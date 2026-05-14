#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys


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


def extract(pattern, text, label):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"missing {label}")
    return match.group(1)


def extract_last(pattern, text, label):
    matches = re.findall(pattern, text)
    if not matches:
        raise ValueError(f"missing {label}")
    return matches[-1]


def build_cmd(args, mode):
    cmd = [
        args.uya,
        "token2wav-flow-probe",
        args.flow_matching_model,
        args.flow_extra_model,
        args.prompt_cache_model,
        args.audio_tokens,
        "--token-limit",
        str(args.token_limit),
        "--block-limit",
        str(args.block_limit),
        "--n-timesteps",
        str(args.n_timesteps),
    ]
    if args.session:
        cmd.append("--session")
    if mode == "embed":
        cmd.append("--embed-mu")
    elif mode == "encoder":
        cmd.extend(["--encoder", args.encoder_model])
    else:
        raise ValueError(f"unsupported mode={mode}")
    return cmd


def parse_probe_output(output):
    mode = extract_last(r"token2wav flow probe: mode=([a-z0-9_-]+)", output, "mode")
    checksum = extract_last(r"checksum_out=0x([0-9a-fA-F]+)", output, "checksum_out")
    frames = int(extract_last(r"(?:emitted_frames|mel_frames)=([0-9]+)", output, "frame_count"))
    return {
        "mode": mode,
        "checksum": checksum.lower(),
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare token2wav surrogate embed-mu output against real encoder-mu output."
    )
    parser.add_argument("--uya", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--flow-matching-model", required=True)
    parser.add_argument("--flow-extra-model", required=True)
    parser.add_argument("--prompt-cache-model", required=True)
    parser.add_argument("--audio-tokens", required=True)
    parser.add_argument("--token-limit", type=int, default=8)
    parser.add_argument("--block-limit", type=int, default=4)
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--session", action="store_true")
    parser.add_argument("--require-different", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    print(
        "token2wav-real-mu config: "
        f"tokens={args.audio_tokens} token_limit={args.token_limit} "
        f"block_limit={args.block_limit} n_timesteps={args.n_timesteps} "
        f"session={1 if args.session else 0}"
    )

    embed_cmd = build_cmd(args, "embed")
    embed_rc, embed_out = run_cmd(embed_cmd, args.timeout)
    print(embed_out, end="")
    if embed_rc != 0:
        print(f"token2wav-real-mu: FAIL embed_rc={embed_rc}")
        return 1

    encoder_cmd = build_cmd(args, "encoder")
    encoder_rc, encoder_out = run_cmd(encoder_cmd, args.timeout)
    print(encoder_out, end="")
    if encoder_rc != 0:
        print(f"token2wav-real-mu: FAIL encoder_rc={encoder_rc}")
        return 1

    try:
        embed = parse_probe_output(embed_out)
        encoder = parse_probe_output(encoder_out)
    except ValueError as exc:
        print(f"token2wav-real-mu: FAIL parse={exc}")
        return 1

    print(
        "token2wav-real-mu summary: "
        f"embed_mode={embed['mode']} embed_checksum=0x{embed['checksum']} "
        f"encoder_mode={encoder['mode']} encoder_checksum=0x{encoder['checksum']} "
        f"embed_frames={embed['frames']} encoder_frames={encoder['frames']}"
    )

    if "encoder" not in encoder["mode"]:
        print(f"token2wav-real-mu: FAIL unexpected encoder mode={encoder['mode']}")
        return 1
    if embed["frames"] != encoder["frames"]:
        print(
            "token2wav-real-mu: FAIL frame mismatch "
            f"embed={embed['frames']} encoder={encoder['frames']}"
        )
        return 1
    if args.require_different and embed["checksum"] == encoder["checksum"]:
        print(
            "token2wav-real-mu: FAIL expected encoder checksum to differ from surrogate embed checksum"
        )
        return 1

    print("token2wav-real-mu: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
