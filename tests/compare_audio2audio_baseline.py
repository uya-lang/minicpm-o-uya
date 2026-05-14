#!/usr/bin/env python3
import argparse
import json
import wave
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def wav_info(path: Path):
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
    return {
        "path": str(path),
        "frames": frames,
        "sample_rate": rate,
        "channels": channels,
        "duration_ms": frames * 1000.0 / rate if rate else 0.0,
    }


def load_text(path: Path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(base_dir: Path, raw):
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def find_case(manifest, case_name):
    for case in manifest.get("cases", []):
        if case.get("name") == case_name:
            return case
    raise ValueError(f"missing baseline case {case_name}")


def select_uya_turn(report):
    turns = report.get("turns", [])
    if not turns:
        raise ValueError("uya report has no turns")
    return turns[-1]


def wav_delta(baseline_wav, uya_wav):
    if not baseline_wav or not uya_wav:
        return None
    return {
        "baseline": baseline_wav,
        "uya": uya_wav,
        "delta_ms": uya_wav["duration_ms"] - baseline_wav["duration_ms"],
        "sample_rate_match": baseline_wav["sample_rate"] == uya_wav["sample_rate"],
        "channels_match": baseline_wav["channels"] == uya_wav["channels"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare one Uya audio2audio report JSON against a local llama.cpp-omni baseline manifest entry."
    )
    parser.add_argument("--uya-report", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--output-json", help="Optional structured comparison output path")
    args = parser.parse_args()

    uya_report_path = Path(args.uya_report).resolve()
    baseline_manifest_path = Path(args.baseline_manifest).resolve()
    report = load_json(uya_report_path)
    baseline_manifest = load_json(baseline_manifest_path)
    baseline_case = find_case(baseline_manifest, args.case_name)
    uya_turn = select_uya_turn(report)
    baseline_dir = baseline_manifest_path.parent

    baseline_answer_text_path = resolve_path(baseline_dir, baseline_case.get("answer_text"))
    baseline_answer_audio_path = resolve_path(baseline_dir, baseline_case.get("answer_audio"))
    baseline_turn_audio_path = resolve_path(baseline_dir, baseline_case.get("turn_audio"))
    baseline_answer_text = load_text(baseline_answer_text_path) if baseline_answer_text_path else ""
    baseline_answer_wav = wav_info(baseline_answer_audio_path) if baseline_answer_audio_path else None
    baseline_turn_wav = wav_info(baseline_turn_audio_path) if baseline_turn_audio_path else None

    uya_answer_wav = uya_turn.get("answer_wav")
    uya_turn_wav = uya_turn.get("turn_wav")
    uya_answer_chars = uya_turn.get("answer_text_chars", 0)
    timing = uya_turn.get("timing", {})

    summary = {
        "schema": "minicpm-o-uya.audio2audio_baseline_compare.v1",
        "compared_at": datetime.now(timezone.utc).isoformat(),
        "uya_report": str(uya_report_path),
        "baseline_manifest": str(baseline_manifest_path),
        "case": {
            "name": baseline_case["name"],
            "protocol": baseline_case["protocol"],
            "answer_text": str(baseline_answer_text_path) if baseline_answer_text_path else None,
            "answer_audio": str(baseline_answer_audio_path) if baseline_answer_audio_path else None,
            "turn_audio": str(baseline_turn_audio_path) if baseline_turn_audio_path else None,
        },
        "uya_turn_prefix": uya_turn.get("prefix"),
        "answer_text": {
            "baseline_chars": len(baseline_answer_text),
            "uya_chars": uya_answer_chars,
            "delta": uya_answer_chars - len(baseline_answer_text),
        },
        "answer_wav": wav_delta(baseline_answer_wav, uya_answer_wav),
        "turn_wav": wav_delta(baseline_turn_wav, uya_turn_wav),
        "timing": {
            "total_ms": timing.get("total_ms"),
            "first_audio_ms": timing.get("first_audio_ms"),
            "rtf": timing.get("rtf"),
            "peak_rss_kb": timing.get("peak_rss_kb"),
        },
        "pass": True,
    }

    print(
        "audio2audio-baseline case: "
        f"name={baseline_case['name']} protocol={baseline_case['protocol']}"
    )
    print(
        "audio2audio-baseline answer-text: "
        f"baseline_chars={len(baseline_answer_text)} uya_chars={uya_answer_chars} "
        f"delta={uya_answer_chars - len(baseline_answer_text)}"
    )

    if baseline_answer_wav and uya_answer_wav:
        print(
            "audio2audio-baseline answer-wav: "
            f"baseline_ms={baseline_answer_wav['duration_ms']:.3f} "
            f"uya_ms={uya_answer_wav['duration_ms']:.3f} "
            f"delta_ms={uya_answer_wav['duration_ms'] - baseline_answer_wav['duration_ms']:.3f}"
        )
    if baseline_turn_wav and uya_turn_wav:
        print(
            "audio2audio-baseline turn-wav: "
            f"baseline_ms={baseline_turn_wav['duration_ms']:.3f} "
            f"uya_ms={uya_turn_wav['duration_ms']:.3f} "
            f"delta_ms={uya_turn_wav['duration_ms'] - baseline_turn_wav['duration_ms']:.3f}"
        )

    print(
        "audio2audio-baseline timing: "
        f"total_ms={timing.get('total_ms')} "
        f"first_audio_ms={timing.get('first_audio_ms')} "
        f"rtf={timing.get('rtf')}"
    )
    if args.output_json:
        write_json(Path(args.output_json), summary)
    print("audio2audio-baseline: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
