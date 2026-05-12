#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys


def run_cmd(cmd, timeout_s):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=timeout_s)
    return proc.returncode, proc.stdout


def extract_uya_text(output):
    marker = "generated text:"
    idx = output.find(marker)
    if idx < 0:
        raise ValueError("missing Uya generated text marker")
    rest = output[idx + len(marker):]
    stop_idx = rest.find("\nstop:")
    if stop_idx >= 0:
        rest = rest[:stop_idx]
    return rest.strip()


def clean_llama_output(output):
    output = output.replace("\r", "\n").replace("\b", "")
    output = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output)
    return output


def extract_llama_text(output):
    output = clean_llama_output(output)
    lines = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("common_perf_print:"):
            break
        lines.append(line)
    skip_prefixes = (
        "main:", "common_", "llama_", "print_info:", "load:", "load_tensors:",
        "sched_", "system_info:", "sampler ", "sampler:", "sampler chain:",
        "generate:", "ggml_", "warning:", "error:", "--", "[",
    )
    candidates = []
    for line in lines:
        if line.startswith(skip_prefixes):
            continue
        if set(line) <= {"."}:
            continue
        if "| memory breakdown" in line:
            continue
        if "=" in line and (":" in line or "MiB" in line):
            continue
        candidates.append(line)
    if not candidates:
        raise ValueError("missing llama generated text candidate")
    return candidates[-1].strip()


def main():
    parser = argparse.ArgumentParser(description="Compare one-token Uya text output with llama.cpp completion output.")
    parser.add_argument("--uya", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--threads", default="1")
    parser.add_argument("--ctx", default="256")
    parser.add_argument("--max-new-tokens", default="1")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    uya_cmd = [
        args.uya, "generate", args.model, args.prompt,
        "--max-new-tokens", args.max_new_tokens,
        "--top-k", "1",
        "--temperature", "1.0",
        "--repeat-penalty", "1.0",
        "--seed", "1",
        "--threads", args.threads,
    ]
    llama_cmd = [
        args.llama,
        "-m", args.model,
        "-p", args.prompt,
        "-n", args.max_new_tokens,
        "--temp", "0",
        "--top-k", "1",
        "--repeat-penalty", "1",
        "--seed", "1",
        "--no-display-prompt",
        "--ctx-size", args.ctx,
        "--threads", args.threads,
        "--no-warmup",
        "--no-perf",
        "-no-cnv",
    ]

    print(f"text-align config: model={args.model} prompt={args.prompt!r} max_new_tokens={args.max_new_tokens} threads={args.threads}")
    rc_uya, out_uya = run_cmd(uya_cmd, args.timeout)
    if rc_uya != 0:
        print(out_uya)
        print(f"text-align: FAIL uya_rc={rc_uya}")
        return 1
    rc_llama, out_llama = run_cmd(llama_cmd, args.timeout)
    if rc_llama != 0:
        print(out_llama)
        print(f"text-align: FAIL llama_rc={rc_llama}")
        return 1
    try:
        uya_text = extract_uya_text(out_uya)
        llama_text = extract_llama_text(out_llama)
    except ValueError as exc:
        print(f"text-align: FAIL parse={exc}")
        print("--- Uya output ---")
        print(out_uya)
        print("--- llama output ---")
        print(out_llama)
        return 1
    print(f"uya generated: {uya_text}")
    print(f"llama generated: {llama_text}")
    if uya_text != llama_text:
        print("text-align: FAIL generated text mismatch")
        return 1
    print("text-align: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
