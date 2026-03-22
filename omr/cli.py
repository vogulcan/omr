from __future__ import annotations

import argparse
from pathlib import Path

from .generator import generate_omr_sheet
from .models import SheetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an A4 OMR sheet PDF.")
    parser.add_argument(
        "--questions",
        required=True,
        help="Comma-separated option counts for each question, for example: 4,4,5,3",
    )
    parser.add_argument(
        "--exam-set-id",
        required=True,
        help="Exam set identifier encoded into the QR payload.",
    )
    parser.add_argument(
        "--variant-id",
        required=True,
        help="Variant identifier encoded into the QR payload.",
    )
    parser.add_argument(
        "--output",
        default="omr-sheet.pdf",
        help="Destination PDF path. Defaults to omr-sheet.pdf",
    )
    parser.add_argument(
        "--title",
        default="Optical Mark Recognition Sheet",
        help="Optional title rendered on the sheet.",
    )
    parser.add_argument(
        "--instructions",
        default="Fill bubbles completely.",
        help="Optional instructions rendered under the title.",
    )
    return parser


def parse_question_counts(raw_value: str) -> list[int]:
    tokens = [token.strip() for token in raw_value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("Question counts must be a non-empty comma-separated list")

    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Question counts must be integers") from exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = SheetConfig(
            question_option_counts=parse_question_counts(args.questions),
            exam_set_id=args.exam_set_id,
            variant_id=args.variant_id,
            title=args.title,
            instructions=args.instructions,
        )
    except ValueError as exc:
        parser.error(str(exc))

    generate_omr_sheet(config, Path(args.output))


if __name__ == "__main__":
    main()
