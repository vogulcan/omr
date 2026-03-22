from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from .grade import BatchGradeResult, GradeResult, grade_directory, grade_path, grade_pdf
from .layout import OPTION_LABELS, PageLayout


@dataclass(slots=True)
class AnnotateResult:
    qr_data: dict | str | None
    student_id: str
    marked_answers: dict[str, list[str]]
    omr_error: str
    annotated_pdf: str


@dataclass(slots=True)
class BatchAnnotateResult:
    source_pdf: str
    qr_data: dict | str | None
    student_id: str
    marked_answers: dict[str, list[str]]
    omr_error: str
    annotated_pdf: str


def annotate_pdf(
    pdf_path: str | Path,
    output_path: str | Path,
    *,
    layout: PageLayout | None = None,
    correct_answers: dict[str, list[str]] | None = None,
) -> AnnotateResult:
    layout = layout or PageLayout()
    source_pdf = Path(pdf_path)
    grade_result = grade_pdf(source_pdf, layout=layout)
    annotated_target = _resolve_annotated_output_path(source_pdf, output_path, directory_mode=False)
    _write_annotated_pdf(
        source_pdf=source_pdf,
        target_pdf=annotated_target,
        layout=layout,
        qr_data=grade_result.qr_data,
        student_id=grade_result.student_id,
        marked_answers=grade_result.marked_answers,
        correct_answers=correct_answers,
        omr_error=grade_result.omr_error,
    )
    return AnnotateResult(
        qr_data=grade_result.qr_data,
        student_id=grade_result.student_id,
        marked_answers=grade_result.marked_answers,
        omr_error=grade_result.omr_error,
        annotated_pdf=str(annotated_target),
    )


def annotate_directory(
    directory: str | Path,
    output_path: str | Path,
    *,
    layout: PageLayout | None = None,
    correct_answers: dict[str, list[str]] | None = None,
) -> list[BatchAnnotateResult]:
    layout = layout or PageLayout()
    target_dir = Path(directory)
    pdf_paths = sorted(path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    results: list[BatchAnnotateResult] = []
    batch_results = {result.source_pdf: result for result in grade_directory(target_dir, layout=layout)}

    for pdf_path in pdf_paths:
        grade_result = batch_results[pdf_path.name]
        annotated_target = _resolve_annotated_output_path(pdf_path, output_path, directory_mode=True)
        _write_annotated_pdf(
            source_pdf=pdf_path,
            target_pdf=annotated_target,
            layout=layout,
            qr_data=grade_result.qr_data,
            student_id=grade_result.student_id,
            marked_answers=grade_result.marked_answers,
            correct_answers=correct_answers,
            omr_error=grade_result.omr_error,
        )
        results.append(
            BatchAnnotateResult(
                source_pdf=pdf_path.name,
                qr_data=grade_result.qr_data,
                student_id=grade_result.student_id,
                marked_answers=grade_result.marked_answers,
                omr_error=grade_result.omr_error,
                annotated_pdf=str(annotated_target),
            )
        )
    return results


def annotate_path(
    path: str | Path,
    output_path: str | Path,
    *,
    layout: PageLayout | None = None,
    correct_answers: dict[str, list[str]] | None = None,
) -> AnnotateResult | list[BatchAnnotateResult]:
    target = Path(path)
    if target.is_dir():
        return annotate_directory(target, output_path, layout=layout, correct_answers=correct_answers)
    return annotate_pdf(target, output_path, layout=layout, correct_answers=correct_answers)


def _resolve_annotated_output_path(source_pdf: Path, output_path: str | Path, *, directory_mode: bool) -> Path:
    target = Path(output_path)
    if directory_mode:
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{source_pdf.stem}-annotated.pdf"

    if target.exists() and target.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{source_pdf.stem}-annotated.pdf"

    if target.suffix.lower() != ".pdf":
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{source_pdf.stem}-annotated.pdf"

    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_annotated_pdf(
    *,
    source_pdf: Path,
    target_pdf: Path,
    layout: PageLayout,
    qr_data: dict | str | None,
    student_id: str,
    marked_answers: dict[str, list[str]],
    correct_answers: dict[str, list[str]] | None,
    omr_error: str,
) -> None:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    first_page = reader.pages[0]
    width = float(first_page.mediabox.width)
    height = float(first_page.mediabox.height)
    overlay_pdf = _build_annotation_overlay(
        page_width=width,
        page_height=height,
        layout=layout,
        qr_data=qr_data,
        student_id=student_id,
        marked_answers=marked_answers,
        correct_answers=correct_answers,
        omr_error=omr_error,
    )
    overlay_reader = PdfReader(io.BytesIO(overlay_pdf))
    for page in reader.pages:
        writer.add_page(page)
    writer.pages[0].merge_page(overlay_reader.pages[0])
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _build_annotation_overlay(
    *,
    page_width: float,
    page_height: float,
    layout: PageLayout,
    qr_data: dict | str | None,
    student_id: str,
    marked_answers: dict[str, list[str]],
    correct_answers: dict[str, list[str]] | None,
    omr_error: str,
) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))

    _draw_metadata_watermark(
        pdf=pdf,
        layout=layout,
        qr_data=qr_data,
        student_id=student_id,
        omr_error=omr_error,
    )
    if correct_answers:
        _draw_correct_answer_overlay(
            pdf=pdf,
            layout=layout,
            correct_answers=correct_answers,
            detected_questions=set(marked_answers),
        )

    pdf.save()
    return buffer.getvalue()


def _draw_metadata_watermark(
    *,
    pdf: canvas.Canvas,
    layout: PageLayout,
    qr_data: dict | str | None,
    student_id: str,
    omr_error: str,
) -> None:
    lines = [f"Student ID: {student_id or '-'}"]
    if isinstance(qr_data, dict):
        lines.append(str(qr_data.get("examSetId", "-")))
        lines.append(str(qr_data.get("variantId", "-")))
    else:
        lines.append(_compact_json(qr_data))
    if omr_error:
        lines.append(f"OMR Error: {omr_error}")

    pdf.saveState()
    pdf.setFont("Helvetica", 9)
    pdf.setFillColor(Color(1, 0, 0, alpha=1))
    text = pdf.beginText()
    text.setTextOrigin(layout.qr_box_left - 42, layout.qr_box_bottom - 24)
    text.setLeading(10)
    for line in lines:
        text.textLine(line)
    pdf.drawText(text)
    pdf.restoreState()


def _draw_correct_answer_overlay(
    *,
    pdf: canvas.Canvas,
    layout: PageLayout,
    correct_answers: dict[str, list[str]],
    detected_questions: set[str],
) -> None:
    pdf.saveState()
    pdf.setFillColor(Color(1, 0, 0, alpha=0.10))
    pdf.setStrokeColor(Color(1, 0, 0, alpha=0.22))
    for question_key, labels in correct_answers.items():
        if question_key not in detected_questions:
            continue
        try:
            question_number = int(question_key)
        except ValueError:
            continue
        if question_number < 1:
            continue
        placement_index = question_number - 1
        column_index = (placement_index % layout.questions_per_page) // layout.questions_per_column
        row_index = placement_index % layout.questions_per_column
        if column_index >= layout.answer_columns_per_page:
            continue
        for label in labels:
            if label not in OPTION_LABELS:
                continue
            option_index = OPTION_LABELS.index(label)
            center_x, center_y = layout.answer_option_center(column_index, row_index, option_index)
            pdf.circle(center_x, center_y, layout.bubble_radius + 2.0, stroke=1, fill=1)
            pdf.setFillColor(Color(0.75, 0, 0, alpha=0.18))
            pdf.setFont("Helvetica-Bold", 8)
            pdf.drawCentredString(center_x, center_y - 2.5, label)
            pdf.setFillColor(Color(1, 0, 0, alpha=0.10))
    pdf.restoreState()


def _compact_json(value: dict | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load_correct_answers(value: str | None) -> dict[str, list[str]] | None:
    if not value:
        return None
    candidate = Path(value)
    raw = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Correct answers must be valid JSON or a path to a JSON file") from exc
    if isinstance(payload, dict) and "answers" in payload and isinstance(payload["answers"], dict):
        payload = payload["answers"]
    if not isinstance(payload, dict):
        raise ValueError("Correct answers JSON must be an object keyed by question number")

    normalized: dict[str, list[str]] = {}
    for question, labels in payload.items():
        question_key = str(question)
        if isinstance(labels, str):
            values = [labels]
        elif isinstance(labels, list) and all(isinstance(label, str) for label in labels):
            values = labels
        else:
            raise ValueError(f"Correct answers for question {question_key} must be a string or list of strings")
        normalized[question_key] = [label.upper() for label in values]
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate a filled OMR sheet PDF or a folder of PDFs.")
    parser.add_argument("path", help="Path to a filled OMR sheet PDF or a directory containing PDF files.")
    parser.add_argument("--output", required=True, help="Annotated PDF output path or directory.")
    parser.add_argument(
        "--correct-answers",
        help="Optional answer key as inline JSON or a path to a JSON file. Correct answers are watermarked in faint red.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        correct_answers = load_correct_answers(args.correct_answers)
    except ValueError as exc:
        parser.error(str(exc))
    result = annotate_path(args.path, args.output, correct_answers=correct_answers)
    if isinstance(result, list):
        payload = [asdict(item) for item in result]
    else:
        payload = asdict(result)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
