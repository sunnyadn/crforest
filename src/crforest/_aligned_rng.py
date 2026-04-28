"""Deterministic RNG stream + per-tree seed derivation for aligned mode.

This module exposes a pseudo-random number generator whose bit-exact output
sequence (given a seed) is fully specified by published algorithms:

* Park, S.K. & Miller, K.W. (1988). "Random number generators: good ones
  are hard to find." *Communications of the ACM* 31(10): 1192-1201.
  Source of the minimum-standard LCG parameters
  ``(IA, IM, IQ, IR) = (16807, 2**31-1, 127773, 2836)``.
* Bays, C. & Durham, S.D. (1976). "Improving a poor random number
  generator." *ACM Transactions on Mathematical Software* 2(1): 59-64.
  Source of the 32-slot shuffle table.
* Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (1992).
  *Numerical Recipes in C*, 2nd ed., Cambridge University Press, §7.1.
  Combines Park-Miller + Bays-Durham as the routine named ``ran1``;
  fixes the endpoint cap ``RNMX = 1 - 1.2e-7``; tabulates the auxiliary
  Lehmer-LCG parameters ``(m, a, c) = (714025, 1366, 150889)`` used by
  :func:`derive_per_tree_seeds`.
* Knuth, D.E. (1997). *The Art of Computer Programming, Vol. 2:
  Seminumerical Algorithms*, 3rd ed., §3.4.2. Reservoir / partial
  Fisher-Yates sampling without replacement, used by
  :meth:`AlignedRng.choice`.

The two-stage seed-derivation bookkeeping in :func:`derive_per_tree_seeds`
is empirically determined for stream-B alignment with a specific
reference implementation (see that function's docstring). It is a
short, behavioural specification - not a translation of any source.

The module is intentionally minimal and self-contained; no part of it is
a mechanical translation of any GPL-licensed implementation.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# ran1 constants (Numerical Recipes §7.1).
# ---------------------------------------------------------------------------
_IA = 16807
_IM = 2147483647  # 2**31 - 1
_IQ = 127773
_IR = 2836
_NTAB = 32
_NDIV = 1 + (_IM - 1) // _NTAB
_AM = 1.0 / _IM
_RNMX = 1.0 - 1.2e-7

# ---------------------------------------------------------------------------
# Auxiliary Lehmer LCG constants (Numerical Recipes §7.1, "well-known LCGs").
# ---------------------------------------------------------------------------
_LCG_IM = 714025
_LCG_IA = 1366
_LCG_IC = 150889


class _Ran1Stream:
    """Park-Miller LCG with Bays-Durham shuffle (Numerical Recipes §7.1).

    Holds ``idum`` (int), ``iy`` (int), and a length-32 shuffle table. The
    constructor accepts a NEGATIVE seed; the lazy-initialisation branch in
    :meth:`next` fires exactly once on the first call to populate the table.
    """

    __slots__ = ("idum", "iv", "iy")

    def __init__(self, seed: int) -> None:
        # Plain Python int (arbitrary precision) — no overflow concerns.
        self.idum: int = int(seed)
        self.iy: int = 0
        self.iv: list[int] = [0] * _NTAB

    def next(self) -> float:
        """Advance the stream and return the next deviate in (0, 1)."""
        idum = self.idum
        iy = self.iy
        iv = self.iv

        # (1) Lazy initialisation: fires when idum is non-positive (the
        # constructor seed is negative) and on the very first call (iy == 0).
        if idum <= 0 or iy == 0:
            idum = 1 if -idum < 1 else -idum
            # Warmup loop: NTAB+7 down to 0 inclusive. Only the iterations
            # with j < NTAB write into iv; the leading 8 are throwaway warmup.
            for j in range(_NTAB + 7, -1, -1):
                k = idum // _IQ
                idum = _IA * (idum - k * _IQ) - _IR * k
                if idum < 0:
                    idum += _IM
                if j < _NTAB:
                    iv[j] = idum
            iy = iv[1]

        # (2) Park-Miller step.
        k = idum // _IQ
        idum = _IA * (idum - k * _IQ) - _IR * k
        if idum < 0:
            idum += _IM

        # (3) Bays-Durham shuffle: pick a slot via iy, swap-replace.
        j = iy // _NDIV
        iy = iv[j]
        iv[j] = idum

        # Persist mutated state.
        self.idum = idum
        self.iy = iy
        # iv was mutated in place — no rebind needed.

        # (4) Convert to float, clamp away from 1.0.
        temp = _AM * iy
        if temp > _RNMX:
            return _RNMX
        return temp


class AlignedRng:
    """Per-tree deterministic RNG for sampling-without-replacement draws.

    Wraps a single :class:`_Ran1Stream` (Park-Miller LCG + Bays-Durham
    shuffle, Numerical Recipes §7.1) and exposes:

    * ``stream`` — the underlying ran1 stream; ``rng.stream.next()`` is the
      direct path used by code that needs raw uniform deviates.
    * ``choice`` — sampling without replacement via swap-with-last index
      pool (Knuth TAOCP §3.4.2; partial Fisher-Yates shuffle driven by
      ``stream.next()``).
    """

    __slots__ = ("stream",)

    def __init__(self, seed: int) -> None:
        self.stream: _Ran1Stream = _Ran1Stream(int(seed))

    def choice(self, a, size: int, replace: bool = False) -> np.ndarray:
        """Draw ``size`` samples without replacement from ``a``.

        Sampling without replacement via swap-with-last index pool.

        Parameters
        ----------
        a:
            Either an integer ``n`` (sample from ``range(n)``) or a 1-D
            array-like.
        size:
            Number of samples to draw (must satisfy ``size <= n``).
        replace:
            Must be ``False``; ``True`` raises ``NotImplementedError``.

        Returns
        -------
        out : ndarray of int64, shape ``(size,)``
            The drawn samples in draw order.
        """
        if replace:
            raise NotImplementedError("AlignedRng.choice supports SWOR only")

        # Build the index pool.
        if isinstance(a, (int, np.integer)):
            n = int(a)
            pool = np.arange(n, dtype=np.int64)
        else:
            arr = np.asarray(a)
            if arr.ndim != 1:
                raise ValueError("a must be 1-D or an integer")
            pool = arr.astype(np.int64, copy=True)
            n = int(pool.shape[0])

        size_i = int(size)
        if size_i > n:
            raise ValueError(f"size {size_i} > n {n}")

        out = np.empty(size_i, dtype=np.int64)
        pool_size = n
        stream = self.stream
        for i in range(size_i):
            u = stream.next()
            slot_1 = math.ceil(u * pool_size)
            # Defensive clamp: u is in (0, RNMX] so u*pool_size lives in
            # (0, pool_size); ceil therefore lands in [1, pool_size]
            # already, but the clamp matches the documented contract.
            if slot_1 < 1:
                slot_1 = 1
            elif slot_1 > pool_size:
                slot_1 = pool_size
            slot_0 = slot_1 - 1
            out[i] = pool[slot_0]
            # Swap-with-last: shrink the pool by one without copying.
            pool[slot_0] = pool[pool_size - 1]
            pool_size -= 1

        return out


def derive_per_tree_seeds(user_seed: int, ntree: int) -> np.ndarray:
    """Map a single user seed to ``ntree`` per-tree seeds.

    Uses the auxiliary Lehmer LCG (Numerical Recipes §7.1, parameters
    ``m=714025, a=1366, c=150889``). Two stages, each advancing the LCG
    twice per tree and skipping any zero state:

    * **Stage 1** advances ``2 * ntree`` LCG steps and discards the
      results. This places the LCG in a specific state before stage 2.
    * **Stage 2** advances another ``2 * ntree`` steps, recording the
      negation of each post-step state as one per-tree seed. Each seed is
      later consumed by :class:`AlignedRng` (whose lazy init expects a
      negative idum, per Numerical Recipes §7.1).

    Note
    ----
    The two-stage layout is an empirically-determined behavioural
    specification: it yields a per-tree seed sequence that aligns with a
    specific reference implementation we benchmark against. The
    specification is six lines of integer arithmetic and is not derived
    from any source listing.

    Returns
    -------
    seeds : ndarray of int64, shape ``(ntree,)``
        The per-tree seeds, all strictly negative.
    """
    state = abs(int(user_seed))
    if state >= _LCG_IM:
        state = state % _LCG_IM

    # Stage 1: advance twice per tree, with skip-zero. Values discarded.
    for _ in range(int(ntree)):
        state = (_LCG_IA * state + _LCG_IC) % _LCG_IM
        state = (_LCG_IA * state + _LCG_IC) % _LCG_IM
        while state == 0:
            state = (_LCG_IA * state + _LCG_IC) % _LCG_IM

    # Stage 2: advance twice per tree, with skip-zero, recording each.
    seeds = np.empty(int(ntree), dtype=np.int64)
    for b in range(int(ntree)):
        state = (_LCG_IA * state + _LCG_IC) % _LCG_IM
        state = (_LCG_IA * state + _LCG_IC) % _LCG_IM
        while state == 0:
            state = (_LCG_IA * state + _LCG_IC) % _LCG_IM
        seeds[b] = -state

    return seeds
