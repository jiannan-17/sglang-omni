#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <venv-path> <cosyvoice-repository> <cosyvoice-commit>" >&2
  exit 1
fi

VENV_PATH="$(cd "$1" && pwd)"
COSYVOICE_REPOSITORY="$2"
COSYVOICE_COMMIT="$3"
COSYVOICE_PATH="${VENV_PATH}/src/CosyVoice"
PYTHON="${VENV_PATH}/bin/python"

git clone --filter=blob:none --no-checkout "${COSYVOICE_REPOSITORY}" "${COSYVOICE_PATH}"
git -C "${COSYVOICE_PATH}" checkout --detach "${COSYVOICE_COMMIT}"
git -C "${COSYVOICE_PATH}" submodule update --init --depth=1 third_party/Matcha-TTS
SITE_PACKAGES="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
# Note (Jiannan Li): TTS jobs replace PYTHONPATH, so persist both source paths in the venv.
printf '%s\n' "${COSYVOICE_PATH}" "${COSYVOICE_PATH}/third_party/Matcha-TTS" \
  > "${SITE_PACKAGES}/cosyvoice.pth"
