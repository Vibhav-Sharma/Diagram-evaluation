#!/usr/bin/env python3
"""CLI entry point for the text-understanding engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract clean semantic text nodes from a messy flowchart image."
    )
    parser.add_argument("image", type=Path, help="Path to flowchart image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: stdout)",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for debug overlays (default: <image_dir>/debug/<stem>)",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip debug visualization outputs",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="Minimum PaddleOCR confidence (default: 0.25)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        return 1

    from text_engine.pipeline import run_text_pipeline

    result = run_text_pipeline(
        args.image,
        debug_dir=args.debug_dir,
        min_ocr_confidence=args.min_confidence,
        save_debug=not args.no_debug,
    )

    payload = result.nodes_as_dicts()
    text = json.dumps(payload, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(payload)} nodes to {args.output}")
    else:
        print(text)

    if result.debug_dir:
        print(f"Debug artifacts: {result.debug_dir}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
