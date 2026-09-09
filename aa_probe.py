#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Concurrency-1 A/A reproducibility probe for the Qwen3-Omni speech server.

record:  python aa_probe.py record --url http://127.0.0.1:8008 --label default-s1 --out default-s1.json \
             [--repeats 2] [--passes 2] [--n 5] [--seed 1234]
compare: python aa_probe.py compare a.json [b.json ...]

`record` sends every prompt `--repeats` times back to back (consecutive A/A on
one server process), then repeats the whole pass `--passes` times.  Each request
records HTTP status, finish_reason, the ordered SSE delta types, the thinker
text, every audio chunk's sample count and SHA-256, and the SHA-256 of the
concatenated PCM.  Requests use temperature 0 (thinker greedy), talker_top_k 1
(talker greedy up to exact ties), a fixed seed, streaming PCM.

`compare` reports, per prompt, for every pair of recordings of that prompt
(consecutive pairs within a file, all pairs across files): text identical?,
chunk counts, first-chunk SHA identical?, leading bitwise-identical chunks,
whole-PCM identical?.  Exit 0 only if every pair is bitwise identical.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import itertools
import json
import sys
import time

DEFAULT_PROMPTS = [
    "Please introduce yourself in two sentences.",
    "Explain in three short sentences why the sky is blue.",
    "Count from one to fifteen, slowly, with a short pause after each number.",
    "Tell me a very short bedtime story about a lighthouse keeper and a seagull.",
    "Describe the taste of a fresh orange to someone who has never had one.",
]


def _one_request(args, prompt_index, rep, pass_index, prompt):
    import requests  # only the recorder needs it

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"format": "pcm"},
        "stream": True,
        "temperature": 0.0,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "talker_top_k": 1,
        "talker_top_p": 1.0,
    }
    rec = {
        "prompt_index": prompt_index, "rep": rep, "pass": pass_index, "prompt": prompt,
        "http_status": None, "error": None, "finish_reason": None, "delta_types": [],
        "text": "", "chunk_samples": [], "chunk_sha256": [], "n_audio_chunks": 0,
        "total_samples": 0, "pcm_sha256": None, "wall_s": None, "t_start": time.time(),
    }
    t0 = time.perf_counter()
    text_parts, pcm = [], bytearray()
    try:
        with requests.post(f"{args.url.rstrip('/')}/v1/chat/completions", json=body, stream=True, timeout=args.timeout) as resp:
            rec["http_status"] = resp.status_code
            if resp.status_code != 200:
                rec["error"] = resp.text[:500]
                return rec
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode() if isinstance(line, bytes) else line
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                event = json.loads(payload)
                if "error" in event:
                    rec["error"] = json.dumps(event["error"])[:500]
                    break
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        rec["finish_reason"] = choice["finish_reason"]
                    if delta.get("content"):
                        rec["delta_types"].append("text")
                        text_parts.append(delta["content"])
                    audio = delta.get("audio")
                    if audio and audio.get("data"):
                        data = base64.b64decode(audio["data"])
                        rec["delta_types"].append("audio")
                        pcm.extend(data)
                        rec["chunk_samples"].append(len(data) // 2)
                        rec["chunk_sha256"].append(hashlib.sha256(data).hexdigest())
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
    rec["wall_s"] = time.perf_counter() - t0
    rec["text"] = "".join(text_parts)
    rec["n_audio_chunks"] = len(rec["chunk_samples"])
    rec["total_samples"] = sum(rec["chunk_samples"])
    rec["pcm_sha256"] = hashlib.sha256(bytes(pcm)).hexdigest()
    return rec


def record(args):
    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as fh:
            prompts = [l.strip() for l in fh if l.strip()]
    prompts = prompts[: args.n] if args.n else prompts
    results = []
    for pass_index in range(args.passes):
        for pi, prompt in enumerate(prompts):
            for rep in range(args.repeats):
                r = _one_request(args, pi, rep, pass_index, prompt)
                results.append(r)
                ok = r["http_status"] == 200 and r["error"] is None and r["finish_reason"]
                print(f"[{args.label}] pass{pass_index} #{pi} rep{rep} {'OK ' if ok else 'FAIL'} "
                      f"status={r['http_status']} finish={r['finish_reason']} chunks={r['n_audio_chunks']} "
                      f"samples={r['total_samples']} first={(r['chunk_sha256'] or ['-'])[0][:10]} "
                      f"pcm={r['pcm_sha256'][:10]} wall={r['wall_s']:.1f}s" + (f" error={r['error']}" if r["error"] else ""), flush=True)
    with open(args.out, "w") as fh:
        json.dump({"label": args.label, "url": args.url, "seed": args.seed, "max_tokens": args.max_tokens,
                   "repeats": args.repeats, "passes": args.passes, "results": results}, fh, indent=1)
    failed = [(r["pass"], r["prompt_index"], r["rep"]) for r in results if not (r["http_status"] == 200 and r["error"] is None)]
    print(f"wrote {args.out}; failed requests: {failed or 'none'}")
    sys.exit(1 if failed else 0)


def _leading(ra, rb):
    n = 0
    for x, y in zip(zip(ra["chunk_samples"], ra["chunk_sha256"]), zip(rb["chunk_samples"], rb["chunk_sha256"])):
        if x != y:
            break
        n += 1
    return n


def _pair_row(kind, la, lb, ra, rb):
    first_same = bool(ra["chunk_sha256"] and rb["chunk_sha256"] and ra["chunk_sha256"][0] == rb["chunk_sha256"][0])
    return {
        "kind": kind, "a": la, "b": lb, "prompt": ra["prompt_index"],
        "text_same": ra["text"] == rb["text"], "chunks": (ra["n_audio_chunks"], rb["n_audio_chunks"]),
        "first_chunk_same": first_same, "leading_identical": _leading(ra, rb),
        "pcm_same": ra["pcm_sha256"] == rb["pcm_sha256"],
        "ok": ra["http_status"] == 200 and rb["http_status"] == 200 and not ra["error"] and not rb["error"],
    }


def compare(args):
    files = [json.load(open(p)) for p in args.files]
    rows = []
    # within-file: consecutive reps (same pass) and cross-pass, per prompt
    for f in files:
        by_prompt = {}
        for r in f["results"]:
            by_prompt.setdefault(r["prompt_index"], []).append(r)
        for pi, recs in sorted(by_prompt.items()):
            for ra, rb in itertools.combinations(recs, 2):
                same_pass = ra["pass"] == rb["pass"]
                kind = "consecutive" if same_pass and abs(ra["rep"] - rb["rep"]) == 1 else ("same-pass" if same_pass else "cross-pass")
                rows.append(_pair_row(f"within:{kind}", f"{f['label']}/p{ra['pass']}r{ra['rep']}", f"{f['label']}/p{rb['pass']}r{rb['rep']}", ra, rb))
    # cross-file: every recording of a prompt in file A vs every in file B
    for fa, fb in itertools.combinations(files, 2):
        for ra in fa["results"]:
            for rb in fb["results"]:
                if ra["prompt_index"] != rb["prompt_index"]:
                    continue
                rows.append(_pair_row("cross-file", f"{fa['label']}/p{ra['pass']}r{ra['rep']}", f"{fb['label']}/p{rb['pass']}r{rb['rep']}", ra, rb))
    all_same = True
    print(f"{'kind':22} {'prompt':>6} {'A':26} {'B':26} {'text':5} {'chunks':>9} {'first':6} {'lead':>5} {'pcm':4}")
    for row in rows:
        all_same &= row["pcm_same"] and row["ok"]
        print(f"{row['kind']:22} {row['prompt']:>6} {row['a']:26} {row['b']:26} {'same' if row['text_same'] else 'DIFF':5} "
              f"{str(row['chunks']):>9} {'same' if row['first_chunk_same'] else 'DIFF':6} {row['leading_identical']:>5} {'same' if row['pcm_same'] else 'DIFF':4}")
    # summary by kind
    print("== summary ==")
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"{kind:22} pairs={len(sub):3d} text_same={sum(r['text_same'] for r in sub):3d} "
              f"first_chunk_same={sum(r['first_chunk_same'] for r in sub):3d} pcm_same={sum(r['pcm_same'] for r in sub):3d} "
              f"zero_leading={sum(1 for r in sub if r['leading_identical']==0):3d}")
    texts = {}
    for f in files:
        for r in f["results"]:
            texts.setdefault(r["prompt_index"], set()).add(r["text"])
    print("distinct thinker texts per prompt across all recordings:", {k: len(v) for k, v in sorted(texts.items())})
    print("PARITY: all pairs bitwise identical" if all_same else "PARITY FAILED: see DIFF rows")
    sys.exit(0 if all_same else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--url", required=True); r.add_argument("--label", required=True); r.add_argument("--out", required=True)
    r.add_argument("--model", default="default"); r.add_argument("--n", type=int, default=0); r.add_argument("--prompts", default=None)
    r.add_argument("--repeats", type=int, default=2); r.add_argument("--passes", type=int, default=2)
    r.add_argument("--seed", type=int, default=1234); r.add_argument("--max-tokens", type=int, default=256); r.add_argument("--timeout", type=float, default=600.0)
    r.set_defaults(func=record)
    c = sub.add_parser("compare"); c.add_argument("files", nargs="+"); c.set_defaults(func=compare)
    a = p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
