#!/usr/bin/env bash
# Convert qwen3_1_7b_merged_v7 → f16 + Q8/Q6/Q5/Q4 on AGH.
set -euo pipefail
AGH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${AGH_DIR}/env.sh"
mkdir -p "${AGH_DIR}/logs"
LOG="${AGH_DIR}/logs/convert_gguf.log"
exec > >(tee -a "${LOG}") 2>&1

cd "${ADTC_ROOT}"
export ADTC_ROOT

# Prefer repo-bundled llama.cpp convert if present; else require CONVERT_HF_TO_GGUF.
LLAMA_SRC="${ADTC_ROOT}/tools/llama.cpp/src-b10451"
LLAMA_BIN="${ADTC_ROOT}/tools/llama.cpp/llama-b10451"
CONVERT="${CONVERT_HF_TO_GGUF:-${LLAMA_SRC}/convert_hf_to_gguf.py}"
OUT_DIR="${ADTC_ROOT}/artifacts/gguf/adapted"
NAME="qwen3_1_7b_merged_v7"
HF_DIR="${ADTC_ROOT}/training/runs/${NAME}"
mkdir -p "${OUT_DIR}" docs/artifacts/v7

[[ -d "${HF_DIR}" ]] || { echo "missing ${HF_DIR}" >&2; exit 1; }

if [[ ! -f "${CONVERT}" ]] || ! command -v llama-quantize >/dev/null 2>&1; then
  echo "[gguf_v7] llama.cpp missing — running setup_llama_cpp.sh"
  bash "${AGH_DIR}/setup_llama_cpp.sh"
  # shellcheck disable=SC1091
  source "${AGH_DIR}/env.sh"
  CONVERT="${CONVERT_HF_TO_GGUF:-${LLAMA_SRC}/convert_hf_to_gguf.py}"
fi
if [[ -d "${LLAMA_BIN}" ]]; then
  export PATH="${LLAMA_BIN}:${PATH}"
  export LD_LIBRARY_PATH="${LLAMA_BIN}:${LD_LIBRARY_PATH:-}"
fi

[[ -f "${CONVERT}" ]] || {
  echo "missing ${CONVERT}" >&2
  echo "  Run: bash ${AGH_DIR}/setup_llama_cpp.sh" >&2
  echo "  Or set CONVERT_HF_TO_GGUF to convert_hf_to_gguf.py" >&2
  exit 1
}
python -m pip install -q 'gguf>=0.10' sentencepiece protobuf || true

F16="${OUT_DIR}/${NAME}-f16.gguf"
if [[ ! -f "${F16}" ]]; then
  echo "[gguf_v7] HF→F16 ${NAME}"
  python "${CONVERT}" "${HF_DIR}" --outfile "${F16}" --outtype f16
else
  echo "[gguf_v7] reuse ${F16}"
fi

if ! command -v llama-quantize >/dev/null 2>&1; then
  echo "error: llama-quantize not on PATH" >&2
  echo "  Run: bash ${AGH_DIR}/setup_llama_cpp.sh" >&2
  exit 1
fi

FAILED=0
for q in Q8_0 Q6_K Q5_K_M Q4_K_M; do
  QOUT="${OUT_DIR}/${NAME}-${q}.gguf"
  if [[ -f "${QOUT}" ]]; then
    echo "[gguf_v7] reuse ${QOUT}"
    continue
  fi
  echo "[gguf_v7] quantize ${q}"
  llama-quantize "${F16}" "${QOUT}" "${q}" || FAILED=$((FAILED + 1))
done

python - <<PY
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone
out_dir = Path("${OUT_DIR}")
name = "${NAME}"
files = {}
for p in sorted(out_dir.glob(f"{name}-*.gguf")):
    files[p.name] = {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
manifest = {
  "name": name,
  "files": files,
  "updated": datetime.now(timezone.utc).isoformat(),
}
path = Path("docs/artifacts/v7/gguf_manifest.json")
path.write_text(json.dumps(manifest, indent=2) + "\n")
print("wrote", path, "n=", len(files))
PY

echo "[gguf_v7] failed=${FAILED}"
exit "${FAILED}"
