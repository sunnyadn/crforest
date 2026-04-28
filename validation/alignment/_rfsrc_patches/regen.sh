#!/usr/bin/env bash
# Rebuild the trace-instrumented randomForestSRC 3.6.2 library used by
# cascade_diagnostic.py / rank_flip_diagnostic.py / grid_mismatch_falsification.py.
#
# Downloads the pristine CRAN tarball, applies the three patches in this
# directory, and installs the patched package into a throwaway library
# location that the diagnostic scripts importr() from.
#
# Defaults match what the diagnostic scripts expect:
#   SRC_DIR = /tmp/rfsrc_instrumented   (patched source tree)
#   LIB_DIR = /tmp/rfsrc_patched_lib    (R library target)
# Override via env vars if needed.

set -euo pipefail

PKG_VERSION="${RFSRC_VERSION:-3.6.2}"
SRC_DIR="${RFSRC_SRC_DIR:-/tmp/rfsrc_instrumented}"
LIB_DIR="${RFSRC_LIB_DIR:-/tmp/rfsrc_patched_lib}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CRAN_URLS=(
  "https://cran.r-project.org/src/contrib/randomForestSRC_${PKG_VERSION}.tar.gz"
  "https://cran.r-project.org/src/contrib/Archive/randomForestSRC/randomForestSRC_${PKG_VERSION}.tar.gz"
)

echo "==> regen.sh for randomForestSRC ${PKG_VERSION}"
echo "    SRC_DIR = ${SRC_DIR}"
echo "    LIB_DIR = ${LIB_DIR}"
echo "    PATCHES = ${PATCH_DIR}"

rm -rf "${SRC_DIR}"
mkdir -p "$(dirname "${SRC_DIR}")" "${LIB_DIR}"

TARBALL="$(mktemp -t rfsrc-XXXXXX.tar.gz)"
trap 'rm -f "${TARBALL}"' EXIT

echo "==> downloading pristine tarball"
downloaded=0
for url in "${CRAN_URLS[@]}"; do
  if curl -fsSL --max-time 120 -o "${TARBALL}" "${url}"; then
    echo "    got ${url}"
    downloaded=1
    break
  fi
done
[[ ${downloaded} -eq 1 ]] || { echo "ERROR: could not fetch tarball from CRAN (current or Archive)"; exit 1; }

echo "==> extracting"
TMPDIR_EXTRACT="$(mktemp -d)"
trap 'rm -f "${TARBALL}"; rm -rf "${TMPDIR_EXTRACT}"' EXIT
tar -xzf "${TARBALL}" -C "${TMPDIR_EXTRACT}"
mv "${TMPDIR_EXTRACT}/randomForestSRC" "${SRC_DIR}"

echo "==> applying patches"
for p in random.c.patch splitSurv.c.patch splitUtil.c.patch importancePerm.c.patch survivalE.c.patch; do
  echo "    ${p}"
  patch -p1 -d "${SRC_DIR}" --quiet < "${PATCH_DIR}/${p}"
done

echo "==> R CMD INSTALL -l ${LIB_DIR} ${SRC_DIR}"
R CMD INSTALL -l "${LIB_DIR}" "${SRC_DIR}"

echo "==> done. Use in R via library(randomForestSRC, lib.loc='${LIB_DIR}')"
echo "    Set RFSRC_TRACE=<path> before fit to capture trace events."
