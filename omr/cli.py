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
        help="Total number of questions to print on the sheet.",
    )
    parser.add_argument(
        "--choices",
        required=True,
        help="Number of choices for every question, for example: 4 or 5",
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


def parse_question_count(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("Question count must be an integer") from exc
    if value < 1:
        raise ValueError("Question count must be at least 1")
    return value


def parse_choice_count(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("Choice count must be an integer") from exc
    if value < 2 or value > 5:
        raise ValueError("Choice count must be between 2 and 5")
    return value


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = SheetConfig(
            question_count=parse_question_count(args.questions),
            choice_count=parse_choice_count(args.choices),
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
