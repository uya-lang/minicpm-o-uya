#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description="Run Uya TTS simplex probe and compare audio token chunks against llama.cpp-omni chunk dumps.")
    parser.add_argument("--uya", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--projector-model", required=True)
    parser.add_argument("--llm-debug-dir", required=True)
    parser.add_argument("--ref-dir", required=True, help="Directory containing audio_tokens_chunk_*.txt from llama.cpp-omni")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-audio-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    temp_dir = Path(tempfile.mkdtemp(prefix="minicpm-o-uya-tts-align-"))
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
        rc, output = run_cmd(cmd, args.timeout)
        print(output, end="")
        if rc != 0:
            print(f"tts-token-align: FAIL uya_rc={rc}")
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
            return 1
        print("tts-token-align: PASS")
        return 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
