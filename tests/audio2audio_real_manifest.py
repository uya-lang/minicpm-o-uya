#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
import wave


def parse_value(text: str):
    if text == "":
        return text
    try:
        if text.startswith(("0x", "-0x")):
            return int(text, 16)
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_timing_log(path: str):
    data = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            data[key] = parse_value(value)
    return data


def wav_info(path: str):
    if not os.path.exists(path):
        return None
    with wave.open(path, "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    duration_ms = 0.0
    if rate > 0:
        duration_ms = frames * 1000.0 / rate
    return {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "sample_rate": rate,
        "channels": channels,
        "sample_width_bytes": width,
        "frames": frames,
        "duration_ms": duration_ms,
    }


def count_glob(pattern: str):
    return len(sorted(glob.glob(pattern)))


def build_turn_entry(artifact_dir: str, timing_path: str):
    timing = parse_timing_log(timing_path)
    timing_name = os.path.basename(timing_path)
    if timing_name == "timing.log":
        prefix = ""
    else:
        prefix = timing_name[: -len("timing.log")]
    answer_txt = os.path.join(artifact_dir, f"{prefix}answer.txt")
    answer_wav = os.path.join(artifact_dir, f"{prefix}answer.wav")
    turn_wav = os.path.join(artifact_dir, f"{prefix}turn.wav")
    llm_chunk_pattern = os.path.join(artifact_dir, f"{prefix}llm_token_ids_chunk_*.txt")
    audio_chunk_pattern = os.path.join(artifact_dir, f"{prefix}audio_tokens_chunk_*.txt")
    wav_chunk_pattern = os.path.join(artifact_dir, f"{prefix}answer_chunk_*.wav")
    answer_text = None
    if os.path.exists(answer_txt):
        with open(answer_txt, "r", encoding="utf-8") as handle:
            answer_text = handle.read()
    return {
        "prefix": prefix[:-1] if prefix.endswith("_") else prefix,
        "timing": timing,
        "timing_path": timing_path,
        "answer_text_path": answer_txt if os.path.exists(answer_txt) else None,
        "answer_text_chars": len(answer_text) if answer_text is not None else 0,
        "answer_wav": wav_info(answer_wav),
        "turn_wav": wav_info(turn_wav),
        "llm_chunk_count": count_glob(llm_chunk_pattern),
        "audio_token_chunk_count": count_glob(audio_chunk_pattern),
        "answer_chunk_wav_count": count_glob(wav_chunk_pattern),
    }


def build_summary(turns):
    def timing_sum(key: str):
        total = 0.0
        for turn in turns:
            value = turn["timing"].get(key)
            if isinstance(value, (int, float)):
                total += float(value)
        return total

    def timing_max(key: str):
        values = []
        for turn in turns:
            value = turn["timing"].get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return max(values) if values else 0.0

    return {
        "turn_count": len(turns),
        "total_generated_tokens": int(timing_sum("generated_tokens")),
        "total_tts_audio_tokens": int(timing_sum("tts_audio_tokens")),
        "total_answer_duration_ms": timing_sum("answer_duration_ms"),
        "total_wall_ms": timing_sum("total_ms"),
        "total_first_audio_ms": timing_sum("first_audio_ms"),
        "max_peak_rss_kb": int(timing_max("peak_rss_kb")),
        "max_rtf": timing_max("rtf"),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize audio2audio-real timing/artifact outputs")
    parser.add_argument("artifact_dir", help="Directory containing timing.log or turn_XXXX_timing.log artifacts")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    artifact_dir = os.path.abspath(args.artifact_dir)
    timing_paths = sorted(glob.glob(os.path.join(artifact_dir, "*timing.log")))
    if not timing_paths:
        print(f"error: no timing logs found under {artifact_dir}", file=sys.stderr)
        return 1

    turns = [build_turn_entry(artifact_dir, path) for path in timing_paths]
    turns.sort(key=lambda item: int(item["timing"].get("turn_index", 0)))
    report = {
        "artifact_dir": artifact_dir,
        "turns": turns,
        "summary": build_summary(turns),
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.write("\n")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
