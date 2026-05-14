#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_baseline(manifest, baseline_name: str):
    for baseline in manifest.get("baselines", []):
        if baseline.get("name") == baseline_name:
            return baseline
    raise ValueError(f"missing tts baseline {baseline_name}")


def load_csv_tokens(path: Path):
    text = path.read_text().strip()
    if not text:
        return []
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()]


def run_cmd(cmd, timeout_s):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout


def compare_one(uya_tokens, ref_tokens):
    prefix = 0
    for a, b in zip(uya_tokens, ref_tokens):
        if a != b:
            break
        prefix += 1
    exact = len(uya_tokens) == len(ref_tokens) and prefix == len(ref_tokens)
    return {
        "uya_count": len(uya_tokens),
        "ref_count": len(ref_tokens),
        "prefix_match": prefix,
        "exact": exact,
    }


def resolve_runtime_config(args):
    baseline = None
    if args.baseline_manifest:
        if not args.baseline_name:
            raise ValueError("--baseline-name is required when --baseline-manifest is set")
        manifest_path = Path(args.baseline_manifest).resolve()
        manifest = load_json(manifest_path)
        baseline = find_baseline(manifest, args.baseline_name)
        if not args.llm_debug_dir:
            args.llm_debug_dir = baseline["llm_debug_dir"]
        if not args.ref_dir:
            args.ref_dir = baseline["ref_dir"]
        if args.count is None:
            args.count = int(baseline.get("chunk_count", 0)) or None
        if args.seed is None and "seed" in baseline:
            args.seed = int(baseline["seed"])
        if baseline.get("greedy"):
            args.greedy = True
        if baseline.get("deterministic_oracle") and not args.allow_inexact:
            args.require_exact = True
        if baseline.get("allow_max_audio_tokens_stop"):
            args.allow_max_audio_tokens_stop = True

    if not args.llm_debug_dir:
        raise ValueError("--llm-debug-dir is required")
    if not args.ref_dir:
        raise ValueError("--ref-dir is required")
    if args.count is None:
        args.count = 1
    if args.seed is None:
        args.seed = 1
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Run Uya TTS simplex probe and compare audio token chunks against llama.cpp-omni chunk dumps.")
    parser.add_argument("--uya", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--projector-model", required=True)
    parser.add_argument("--llm-debug-dir")
    parser.add_argument("--ref-dir", help="Directory containing audio_tokens_chunk_*.txt from llama.cpp-omni")
    parser.add_argument("--baseline-manifest", help="Optional local seeded-baseline manifest JSON")
    parser.add_argument("--baseline-name", help="Named baseline entry inside --baseline-manifest")
    parser.add_argument("--count", type=int)
    parser.add_argument("--max-audio-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--allow-inexact", action="store_true", help="Allow exploratory prefix-only compare even for deterministic manifests")
    parser.add_argument("--allow-max-audio-tokens-stop", action="store_true", help="Treat token-cap truncation as a valid stop for prefix-oracle comparisons")
    parser.add_argument("--output-json", help="Optional JSON summary output path")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    try:
        baseline = resolve_runtime_config(args)
    except ValueError as exc:
        print(f"tts-token-align: FAIL {exc}")
        return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="minicpm-o-uya-tts-align-"))
    summary = {
        "schema": "minicpm-o-uya.tts_token_alignment.v1",
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "uya": str(Path(args.uya).resolve()),
        "tts_model": str(Path(args.tts_model).resolve()),
        "projector_model": str(Path(args.projector_model).resolve()),
        "llm_debug_dir": str(Path(args.llm_debug_dir).resolve()),
        "ref_dir": str(Path(args.ref_dir).resolve()),
        "baseline_name": baseline.get("name") if baseline else None,
        "seed": args.seed,
        "greedy": bool(args.greedy),
        "require_exact": bool(args.require_exact),
        "allow_max_audio_tokens_stop": bool(args.allow_max_audio_tokens_stop),
        "count": args.count,
        "max_audio_tokens": args.max_audio_tokens,
        "chunks": [],
        "pass": False,
    }
    try:
        cmd = [
            args.uya,
            "tts-simplex-probe",
            args.tts_model,
            args.projector_model,
            args.llm_debug_dir,
            "--count",
            str(args.count),
            "--out-dir",
            str(temp_dir),
            "--max-audio-tokens",
            str(args.max_audio_tokens),
            "--seed",
            str(args.seed),
        ]
        if args.greedy:
            cmd.append("--greedy")
        if args.allow_max_audio_tokens_stop:
            cmd.append("--allow-max-audio-tokens-stop")
        rc, output = run_cmd(cmd, args.timeout)
        print(output, end="")
        if rc != 0:
            print(f"tts-token-align: FAIL uya_rc={rc}")
            summary["uya_rc"] = rc
            if args.output_json:
                write_json(Path(args.output_json), summary)
            return 1

        any_mismatch = False
        for idx in range(args.count):
            uya_path = temp_dir / f"audio_tokens_chunk_{idx}.txt"
            ref_path = Path(args.ref_dir) / f"audio_tokens_chunk_{idx}.txt"
            if not uya_path.exists():
                print(f"tts-token-align: FAIL missing uya chunk file {uya_path}")
                return 1
            if not ref_path.exists():
                print(f"tts-token-align: FAIL missing reference chunk file {ref_path}")
                return 1
            uya_tokens = load_csv_tokens(uya_path)
            ref_tokens = load_csv_tokens(ref_path)
            result = compare_one(uya_tokens, ref_tokens)
            result["chunk"] = idx
            result["uya_path"] = str(uya_path)
            result["ref_path"] = str(ref_path)
            summary["chunks"].append(result)
            print(
                "tts-token-align chunk="
                f"{idx} ref={result['ref_count']} uya={result['uya_count']} "
                f"prefix_match={result['prefix_match']} exact={1 if result['exact'] else 0}"
            )
            if not result["exact"]:
                any_mismatch = True
                if uya_tokens and ref_tokens:
                    print(
                        f"tts-token-align chunk={idx} first_uya={uya_tokens[:10]} first_ref={ref_tokens[:10]}"
                    )
        if args.require_exact and any_mismatch:
            print("tts-token-align: FAIL exact mismatch")
            if args.output_json:
                write_json(Path(args.output_json), summary)
            return 1
        summary["pass"] = True
        if args.output_json:
            write_json(Path(args.output_json), summary)
        print("tts-token-align: PASS")
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
