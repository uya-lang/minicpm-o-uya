#!/usr/bin/env python3
import argparse
import glob
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_path(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_csv_tokens(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()]


def build_entry(args):
    llm_debug_dir = Path(args.llm_debug_dir).resolve()
    ref_dir = Path(args.ref_dir).resolve()
    token_paths = sorted(Path(path).resolve() for path in glob.glob(str(ref_dir / "audio_tokens_chunk_*.txt")))
    llm_chunk_dirs = sorted(path.resolve() for path in llm_debug_dir.glob("chunk_*") if path.is_dir())
    if not token_paths:
        raise ValueError(f"no audio_tokens_chunk_*.txt found under {ref_dir}")
    if not llm_chunk_dirs:
        raise ValueError(f"no chunk_* dirs found under {llm_debug_dir}")

    token_files = []
    for index, path in enumerate(token_paths):
        tokens = load_csv_tokens(path)
        token_files.append(
            {
                "chunk": index,
                "path": str(path),
                "token_count": len(tokens),
                "sha256": sha256_path(path),
            }
        )

    return {
        "name": args.baseline_name,
        "source": args.source,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "llm_debug_dir": str(llm_debug_dir),
        "ref_dir": str(ref_dir),
        "llm_chunk_count": len(llm_chunk_dirs),
        "chunk_count": len(token_files),
        "seed": args.seed,
        "temperature": args.temperature,
        "greedy": bool(args.greedy),
        "deterministic_oracle": bool(args.deterministic_oracle or args.greedy),
        "allow_max_audio_tokens_stop": bool(args.allow_max_audio_tokens_stop),
        "command": args.command,
        "notes": args.notes,
        "token_files": token_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Record a local seeded llama.cpp-omni TTS token baseline manifest entry.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--llm-debug-dir", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--deterministic-oracle", action="store_true")
    parser.add_argument("--allow-max-audio-tokens-stop", action="store_true")
    parser.add_argument("--source", default="llama.cpp-omni")
    parser.add_argument("--command")
    parser.add_argument("--notes")
    args = parser.parse_args()

    manifest_path = Path(args.output_manifest).resolve()
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {
            "schema": "minicpm-o-uya.external_tts_token_baselines.v1",
            "note": "Local-only manifest; outputs/ is ignored and this file is not intended for git.",
            "baselines": [],
        }

    entry = build_entry(args)
    baselines = manifest.setdefault("baselines", [])
    updated = False
    for index, baseline in enumerate(baselines):
        if baseline.get("name") == entry["name"]:
            baselines[index] = entry
            updated = True
            break
    if not updated:
        baselines.append(entry)
    baselines.sort(key=lambda item: item.get("name", ""))
    write_json(manifest_path, manifest)

    print(
        "tts-seeded-baseline-recorded: "
        f"name={entry['name']} chunks={entry['chunk_count']} seed={entry['seed']} "
        f"greedy={1 if entry['greedy'] else 0} deterministic={1 if entry['deterministic_oracle'] else 0}"
    )
    print(f"tts-seeded-baseline-manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
