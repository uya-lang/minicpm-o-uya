#!/usr/bin/env python3
import argparse
import struct
import subprocess
import sys
from pathlib import Path


def load_bin(path: Path):
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"short embedding dump: {path}")
    n_tokens, n_embd = struct.unpack_from("<II", data, 0)
    need = 8 + n_tokens * n_embd * 4
    if len(data) != need:
        raise ValueError(f"bad embedding dump size: {path} got={len(data)} expected={need}")
    values = struct.unpack_from(f"<{n_tokens * n_embd}f", data, 8)
    return n_tokens, n_embd, values


def fnv1a_f32(values):
    h = 2166136261
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        for shift in (0, 8, 16, 24):
            h ^= (bits >> shift) & 0xFF
            h = (h * 16777619) & 0xFFFFFFFF
    return h


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


def main():
    parser = argparse.ArgumentParser(description="Compare Uya TTS merged embeddings against llama.cpp-omni merged_embeddings.bin.")
    parser.add_argument("--uya", required=True)
    parser.add_argument("--tts-model", required=True)
    parser.add_argument("--projector-model", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--hidden-bin", required=True)
    parser.add_argument("--llama-merged-bin", required=True)
    parser.add_argument("--audio-bos-id", default="151687")
    parser.add_argument("--max-abs-threshold", type=float, default=1e-5)
    parser.add_argument("--mean-abs-threshold", type=float, default=1e-6)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    uya_out = Path("/tmp/minicpm-o-uya-tts-merge-align.bin")
    if uya_out.exists():
        uya_out.unlink()

    uya_cmd = [
        args.uya,
        "tts-condition-probe",
        args.tts_model,
        args.projector_model,
        args.token_ids,
        args.hidden_bin,
        "--audio-bos-id",
        args.audio_bos_id,
        "--dump-merged",
        str(uya_out),
    ]
    rc, output = run_cmd(uya_cmd, args.timeout)
    print(output, end="")
    if rc != 0:
        print(f"tts-merge-align: FAIL uya_rc={rc}")
        return 1

    try:
        uya_tokens, uya_embd, uya_values = load_bin(uya_out)
        ref_tokens, ref_embd, ref_values = load_bin(Path(args.llama_merged_bin))
    except ValueError as exc:
        print(f"tts-merge-align: FAIL parse={exc}")
        return 1

    print(
        f"tts-merge-align shape: uya=[{uya_tokens},{uya_embd}] "
        f"llama=[{ref_tokens},{ref_embd}]"
    )
    print(
        f"tts-merge-align checksum: uya=0x{fnv1a_f32(uya_values):08x} "
        f"llama=0x{fnv1a_f32(ref_values):08x}"
    )

    if (uya_tokens, uya_embd) != (ref_tokens, ref_embd):
        print("tts-merge-align: FAIL shape mismatch")
        return 1

    errs = [abs(a - b) for a, b in zip(uya_values, ref_values)]
    mean_abs = sum(errs) / len(errs) if errs else 0.0
    max_abs = max(errs) if errs else 0.0
    l1 = sum(errs)
    print(f"tts-merge-align diff: L1={l1:.9f} mean_abs={mean_abs:.9f} max_abs={max_abs:.9f}")
    if max_abs > args.max_abs_threshold or mean_abs > args.mean_abs_threshold:
        print(
            "tts-merge-align: FAIL "
            f"thresholds max_abs<={args.max_abs_threshold} mean_abs<={args.mean_abs_threshold}"
        )
        return 1
    print("tts-merge-align: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
