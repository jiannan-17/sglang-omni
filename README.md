# Qwen3-Omni concurrency-1 audio reproducibility MRE

Companion to the bug issue "Qwen3-Omni: identical greedy requests produce different audio at concurrency 1".
Measured 2026-09-09 on `sgl-project/sglang-omni` main `76a2a48f` (SGLang 0.5.19, FlashInfer 0.6.18), 1x H100 80GB, Runpod.

## Run

```bash
# server (arm "default"); the other arms add one of:
#   --thinker.engine.disable_radix_cache true
#   --talker_ar.engine.enable_deterministic_inference true
#   --thinker.engine.enable_deterministic_inference true
sgl-omni serve --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct --port 8008 \
  --config examples/configs/qwen3_omni_colocated_h100_bf16.yaml --colocate

# client: 5 prompts x 2 consecutive requests x 2 passes, greedy thinker + greedy talker, streaming PCM, per-chunk SHA-256
python aa_probe.py record --url http://127.0.0.1:8008 --label default-s1 --out default-s1.json --repeats 2 --passes 2
# restart the same server command, record again as default-s2, then:
python aa_probe.py compare default-s1.json default-s2.json
```

`repro_main_runbook.sh` is the script that ran all five arms (env capture, checkout, weight download, 2 server starts per arm, compare).

## What is here

- `aa_probe.py` - recorder / comparator (stdlib + `requests`).
- `results/<arm>-s1.prompt1.json` - raw recordings of prompt #1 ("Explain in three short sentences why the sky is blue.") for the default arm and the talker-flag arm: 4 requests each (2 consecutive x 2 passes) with per-chunk sample counts and SHA-256.
- `results/compare-<arm>.txt` - full pairwise comparison output for every arm (8 recordings per prompt across 2 server starts).
- `results/server-<arm>.cmd` - exact server command per arm.
- `logs/server-{default,det_talker_only}-s1.excerpt.log` - merged config dump plus backend / CUDA-graph / prefill (`#cached-token`) / deterministic-mode lines from the first server start of each arm; IPs and socket paths redacted.
- `logs/env.txt` - versions.

## Result in one line

Default kernels: 140/140 same-prompt pairs differ from PCM chunk 0 (20/20 consecutive), thinker text identical, 3/80 requests ran to the 4096-codec-frame cap. `--talker_ar.engine.enable_deterministic_inference true`: 0/140 differ (bitwise identical across requests and server starts), text unchanged. Thinker-only flag: still 140/140 and the thinker's greedy text changes on 3 of 5 prompts.
