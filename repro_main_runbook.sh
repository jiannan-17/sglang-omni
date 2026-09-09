#!/usr/bin/env bash
# Qwen3-Omni c1 A/A reproducibility on CURRENT MAIN (for the #1018 follow-up bug issue).
# Runs ON THE POD (template sglang-omni-base-dev). Everything lands under /workspace/repro.
#   bash repro_main_runbook.sh env      # capture env, checkout main@SHA, FlashInfer cache link, start weight download
#   bash repro_main_runbook.sh run      # 3 configs x 2 server starts x (5 prompts x 2 consecutive reps x 2 passes)
#   bash repro_main_runbook.sh compare  # per-config comparisons
#   bash repro_main_runbook.sh all
set -u
export PATH=/opt/sglang/bin:$PATH
ROOT=/workspace/repro
TOOLS=$ROOT/tools
REPO=/workspace/sglang-omni
SHA=${SHA:-76a2a48f}                       # main @ 2026-09-09 (#2040 SGLang 0.5.19)
MODEL_ID=Qwen/Qwen3-Omni-30B-A3B-Instruct
MODEL_DIR=/workspace/models/Qwen3-Omni-30B-A3B-Instruct
PORT=${PORT:-8008}
CONFIGS=${CONFIGS:-default radix_off det}
STARTS=${STARTS:-2}
mkdir -p "$ROOT" "$TOOLS" /workspace/models
stamp() { date -u +%FT%TZ; }
say() { echo "[$(stamp)] $*"; }

# interpreter that sgl-omni runs with (the 0.5.19 image installs the stack into /opt/sglang)
SGL_BIN=$(command -v sgl-omni || true)
if [ -n "$SGL_BIN" ] && head -1 "$SGL_BIN" | grep -q '^#!'; then PY=$(head -1 "$SGL_BIN" | sed 's/^#!//' | awk '{print $1}'); else PY=$(command -v python3); fi

serve_args_for() {
  case "$1" in
    default)   echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate" ;;
    radix_off) echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate --thinker.engine.disable_radix_cache true" ;;
    det)       echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate --thinker.engine.enable_deterministic_inference true --talker_ar.engine.enable_deterministic_inference true" ;;
    det_talker_only)  echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate --talker_ar.engine.enable_deterministic_inference true" ;;
    det_thinker_only) echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate --thinker.engine.enable_deterministic_inference true" ;;
    det_radix_off) echo "--config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate --thinker.engine.disable_radix_cache true --thinker.engine.enable_deterministic_inference true --talker_ar.engine.enable_deterministic_inference true" ;;
    *) echo "unknown config $1" >&2; return 1 ;;
  esac
}

cmd_env() {
  {
    echo "== env $(stamp) =="
    hostname; nvidia-smi -L
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    nproc; free -g | sed -n 2p; df -h / /workspace 2>/dev/null | tail -2
    echo "sgl-omni: $SGL_BIN ; python: $PY"
    $PY --version
    $PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'devices',torch.cuda.device_count())"
    $PY -m pip show sglang-omni sglang flashinfer-python flashinfer_python sgl-kernel triton 2>/dev/null | grep -E "^(Name|Version|Location|Editable)"
    $PY -c "import sglang_omni;print('sglang_omni from',sglang_omni.__file__)"
    (cd "$REPO" && echo "repo HEAD $(git rev-parse --short HEAD) $(git log -1 --format=%cI)" && git remote -v | head -2)
    ls -d /root/.cache/flashinfer* /root/.cache/sglang/.cache/flashinfer* /opt/sglang 2>/dev/null
    env | grep -i -E "flashinfer|sglang" | sort
  } 2>&1 | tee "$ROOT/env.txt"

  say "checkout main@$SHA in $REPO"
  cd "$REPO" || exit 1
  git fetch --no-tags origin main 2>&1 | tail -1
  git checkout -q --detach "$SHA" && echo "checked out $(git rev-parse --short HEAD) $(git log -1 --format='%cI %s')" | tee -a "$ROOT/env.txt"
  $PY -c "import sglang_omni,os;p=sglang_omni.__file__;print('import resolves to',p); assert p.startswith('$REPO/'), 'NOT the editable checkout'" \
    || { say "sglang_omni not imported from $REPO; installing editable (--no-deps)"; $PY -m pip install -e "$REPO" --no-deps -q 2>&1 | tail -2; }
  $PY -m pip show sglang 2>/dev/null | grep -E "^Version" | tee -a "$ROOT/env.txt"

  # FlashInfer JIT cache: sglang redirects FlashInfer's workspace under ~/.cache/sglang/.cache/flashinfer while the
  # image's prebuilt kernels live in ~/.cache/flashinfer. Link if the prebuilt dir exists and the workspace is absent.
  if [ -d /root/.cache/flashinfer ] && [ ! -e /root/.cache/sglang/.cache/flashinfer ]; then
    mkdir -p /root/.cache/sglang/.cache && ln -s /root/.cache/flashinfer /root/.cache/sglang/.cache/flashinfer && say "linked FlashInfer prebuilt cache"
  else
    say "flashinfer cache dirs: $(ls -d /root/.cache/flashinfer* /root/.cache/sglang/.cache/flashinfer* 2>/dev/null | tr '\n' ' ')"
  fi

  if [ ! -f "$MODEL_DIR/config.json" ]; then
    say "starting weight download $MODEL_ID -> $MODEL_DIR (background, log $ROOT/hf-download.log)"
    ( if command -v hf >/dev/null; then hf download "$MODEL_ID" --local-dir "$MODEL_DIR"; else huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"; fi
      echo "download exit=$?" ) > "$ROOT/hf-download.log" 2>&1 &
    echo $! > "$ROOT/hf-download.pid"
  else
    say "weights already present at $MODEL_DIR"
  fi
}

wait_ready() {
  local pid=$1 log=$2 deadline=$((SECONDS + 1800))
  while [ $SECONDS -lt $deadline ]; do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server died; tail of $log:"; tail -40 "$log"; return 1; fi
    sleep 5
  done
  echo "server not ready after 30 min; tail:"; tail -40 "$log"; return 1
}

run_start() {   # <config> <start-index>
  local cfg=$1 s=$2 label="$cfg-s$s" args
  args=$(serve_args_for "$cfg") || return 1
  local log=$ROOT/$cfg/server-s$s.log
  mkdir -p "$ROOT/$cfg"
  say "== $label: sgl-omni serve --model-path $MODEL_DIR --port $PORT $args =="
  echo "sgl-omni serve --model-path $MODEL_DIR --port $PORT $args" > "$ROOT/$cfg/server-s$s.cmd"
  ( cd "$REPO" && sgl-omni serve --model-path "$MODEL_DIR" --port "$PORT" $args ) > "$log" 2>&1 &
  local pid=$!
  if ! wait_ready "$pid" "$log"; then kill "$pid" 2>/dev/null; return 1; fi
  say "server ready (pid $pid); recording $label"
  $PY "$TOOLS/aa_probe.py" record --url "http://127.0.0.1:$PORT" --label "$label" --out "$ROOT/$cfg/$label.json" --repeats 2 --passes 2 2>&1 | tee "$ROOT/$cfg/$label.record.log"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 5
  pkill -f "[s]glang_omni" 2>/dev/null; sleep 3
  echo "-- error scan --"; grep -nE "CUDA error|cudaError|illegal memory|Traceback|RuntimeError" "$log" | head -10 || true
  echo "-- cached-token lines (thinker prefill) --"; grep -n -i "cached-token\|cached_token\|cache hit\|prefix" "$log" | grep -v -i "graph\|warmup\|capture" | head -30 > "$ROOT/$cfg/server-s$s.cached-token.txt"; wc -l < "$ROOT/$cfg/server-s$s.cached-token.txt"
}

cmd_run() {
  cd "$REPO" && git checkout -q --detach "$SHA"
  if [ -f "$ROOT/hf-download.pid" ] && kill -0 "$(cat "$ROOT/hf-download.pid")" 2>/dev/null; then
    say "waiting for weight download"; while kill -0 "$(cat "$ROOT/hf-download.pid")" 2>/dev/null; do sleep 20; done
  fi
  tail -2 "$ROOT/hf-download.log" 2>/dev/null
  [ -f "$MODEL_DIR/config.json" ] || { say "weights missing"; return 1; }
  say "== run @ $(git rev-parse --short HEAD) configs=[$CONFIGS] starts=$STARTS =="
  for cfg in $CONFIGS; do
    for s in $(seq 1 "$STARTS"); do
      run_start "$cfg" "$s" || say "FAILED $cfg s$s (continuing)"
    done
  done
  say "run done"
}

cmd_compare() {
  for cfg in $CONFIGS; do
    [ -d "$ROOT/$cfg" ] || continue
    say "== compare $cfg =="
    $PY "$TOOLS/aa_probe.py" compare "$ROOT/$cfg"/$cfg-s*.json 2>&1 | tee "$ROOT/$cfg/compare.txt"
  done
  say "== cross-config: default vs det (text/frames) =="
  $PY - "$ROOT" <<'EOF'
import json,glob,sys,os
root=sys.argv[1]
def load(c):
    out=[]
    for f in sorted(glob.glob(f"{root}/{c}/{c}-s*.json")): out+=json.load(open(f))["results"]
    return out
have={c:load(c) for c in ("default","radix_off","det","det_radix_off","det_talker_only","det_thinker_only") if glob.glob(f"{root}/{c}/{c}-s*.json")}
for c,recs in have.items():
    by={}
    for r in recs: by.setdefault(r["prompt_index"],[]).append(r)
    print(f"{c}: per prompt distinct texts / chunk counts / distinct pcm:", {p:(len({r['text'] for r in v}), sorted(r['n_audio_chunks'] for r in v), len({r['pcm_sha256'] for r in v})) for p,v in sorted(by.items())})
if "default" in have and "det" in have:
    d0={r["prompt_index"]:r["text"] for r in have["default"]}
    for r in have["det"]:
        if r["rep"]==0 and r["pass"]==0: print(f"prompt {r['prompt_index']}: det text == default text? {r['text']==d0.get(r['prompt_index'])}")
EOF
}

case "${1:-all}" in
  env) cmd_env ;;
  run) cmd_run ;;
  compare) cmd_compare ;;
  all) cmd_env; cmd_run; cmd_compare; say "ALL DONE" ;;
  *) sed -n '2,8p' "$0" ;;
esac
