#!/usr/bin/env bash
# Idempotent download of TebebAI GGUF from Hugging Face (no credentials).
# Path must match metadata.json → _runtime.model_path
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${ROOT}/model/tebeb_tutor_1.7b.gguf"
URL="https://huggingface.co/nz2212/tebeb_tutor_1.7b/resolve/main/tebeb_tutor_1.7b.gguf"
EXPECTED_BYTES=1257879232

mkdir -p "${ROOT}/model"

if [[ -f "${DEST}" ]]; then
  size="$(stat -c%s "${DEST}" 2>/dev/null || stat -f%z "${DEST}")"
  if [[ "${size}" -eq "${EXPECTED_BYTES}" ]]; then
    echo "[download_model] already present: ${DEST} (${size} bytes)"
    exit 0
  fi
  echo "[download_model] unexpected size ${size} (want ${EXPECTED_BYTES}); re-downloading"
  rm -f "${DEST}"
fi

echo "[download_model] fetching ${URL}"
tmp="${DEST}.partial"
rm -f "${tmp}"
curl -L --fail --retry 5 --retry-delay 2 -o "${tmp}" "${URL}"
size="$(stat -c%s "${tmp}" 2>/dev/null || stat -f%z "${tmp}")"
if [[ "${size}" -ne "${EXPECTED_BYTES}" ]]; then
  echo "[download_model] ERROR: got ${size} bytes, expected ${EXPECTED_BYTES}" >&2
  rm -f "${tmp}"
  exit 1
fi
mv "${tmp}" "${DEST}"
echo "[download_model] done: ${DEST} (${size} bytes)"
