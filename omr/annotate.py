from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color
from reportlab.pdfgen import canvas

from .grade import GradeResult, _AlignedSheet, _grade_pdf_with_alignment
from .layout import OPTION_LABELS, PageLayout
from .pdf_fonts import PdfFontSet, get_pdf_fonts

HOUGH_REFINEMENT_JITTER_RATIO = 0.30
HOUGH_REFINEMENT_MIN_JITTER_PX = 1.5
ANSWER_ROW_REFINEMENT_X_PADDING_RATIO = 2.2
ANSWER_ROW_REFINEMENT_Y_PADDING_RATIO = 1.8
ANSWER_ROW_BUBBLE_CLUSTER_MIN_SPAN_RATIO = 0.9
ANSWER_ROW_BUBBLE_CLUSTER_SPLIT_RATIO = 0.45
ANSWER_ROW_CANDIDATE_MERGE_RATIO = 0.9


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
    grade_result, alignment = _grade_pdf_with_alignment(source_pdf, layout=layout)
    annotated_target = _resolve_annotated_output_path(source_pdf, output_path, directory_mode=False)
    _write_annotated_pdf(
        source_pdf=source_pdf,
        target_pdf=annotated_target,
        layout=layout,
        alignment=alignment,
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

    for pdf_path in pdf_paths:
        try:
            grade_result, alignment = _grade_pdf_with_alignment(pdf_path, layout=layout)
        except Exception as exc:
            grade_result = GradeResult(
                qr_data=None,
                student_id="",
                marked_answers={},
                omr_error=str(exc),
            )
            alignment = None
        annotated_target = _resolve_annotated_output_path(pdf_path, output_path, directory_mode=True)
        _write_annotated_pdf(
            source_pdf=pdf_path,
            target_pdf=annotated_target,
            layout=layout,
            alignment=alignment,
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
    alignment: _AlignedSheet | None,
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
        alignment=alignment,
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
    alignment: _AlignedSheet | None,
    qr_data: dict | str | None,
    student_id: str,
    marked_answers: dict[str, list[str]],
    correct_answers: dict[str, list[str]] | None,
    omr_error: str,
) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(page_width, page_height))
    fonts = get_pdf_fonts()

    _draw_metadata_watermark(
        pdf=pdf,
        layout=layout,
        fonts=fonts,
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
            alignment=alignment,
            page_width=page_width,
            page_height=page_height,
        )

    pdf.save()
    return buffer.getvalue()


def _draw_metadata_watermark(
    *,
    pdf: canvas.Canvas,
    layout: PageLayout,
    fonts: PdfFontSet,
    qr_data: dict | str | None,
    student_id: str,
    omr_error: str,
) -> None:
    lines = [f"Student ID: {student_id or '-'}"]
    if isinstance(qr_data, dict):
        lines.append(f"Exam Set: {qr_data.get('examSetId', '-')}")
        lines.append(f"Variant: {qr_data.get('variantId', '-')}")
    else:
        lines.append(f"QR Data: {_compact_json(qr_data)}")
    if omr_error:
        lines.append(f"OMR Error: {omr_error}")

    left = layout.annotation_box_left
    right = layout.annotation_box_right
    bottom = layout.annotation_box_bottom_y
    top = layout.annotation_box_top_y
    if right <= left or top <= bottom:
        return

    width = right - left
    height = top - bottom
    font_name = fonts.regular
    font_size = 8.5
    leading = 10.0
    wrapped_lines = _wrap_annotation_lines(pdf, lines, font_name, font_size, width - 18.0)

    pdf.saveState()
    pdf.setFillColor(Color(1, 1, 1, alpha=0.84))
    pdf.setStrokeColor(Color(0.72, 0, 0, alpha=0.30))
    pdf.roundRect(left, bottom, width, height, 6, stroke=1, fill=1)
    pdf.setFont(font_name, font_size)
    pdf.setFillColor(Color(0.78, 0, 0, alpha=1))
    text = pdf.beginText()
    text.setTextOrigin(left + 9, top - 14)
    text.setLeading(leading)
    for line in wrapped_lines:
        text.textLine(line)
    pdf.drawText(text)
    pdf.restoreState()


def _draw_correct_answer_overlay(
    *,
    pdf: canvas.Canvas,
    layout: PageLayout,
    correct_answers: dict[str, list[str]],
    detected_questions: set[str],
    alignment: _AlignedSheet | None,
    page_width: float,
    page_height: float,
) -> None:
    bubble_radius = _overlay_bubble_radius(layout, page_width, page_height)

    pdf.saveState()
    pdf.setFillColor(Color(1, 0, 0, alpha=0.14))
    pdf.setStrokeColor(Color(1, 0, 0, alpha=0.28))
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
            if alignment is not None:
                source_x, source_y = _layout_point_to_source_raster_point(
                    layout=layout,
                    alignment=alignment,
                    center_x=center_x,
                    center_y=center_y,
                )
                source_x, source_y = _refine_source_answer_option_center(
                    layout=layout,
                    alignment=alignment,
                    column_index=column_index,
                    row_index=row_index,
                    option_index=option_index,
                    center_x=source_x,
                    center_y=source_y,
                )
                center_x, center_y = _source_raster_point_to_pdf_point(
                    alignment=alignment,
                    page_width=page_width,
                    page_height=page_height,
                    center_x=source_x,
                    center_y=source_y,
                )
            pdf.circle(center_x, center_y, bubble_radius, stroke=1, fill=1)
    pdf.restoreState()


def _layout_point_to_source_raster_point(
    *,
    layout: PageLayout,
    alignment: _AlignedSheet,
    center_x: float,
    center_y: float,
) -> tuple[float, float]:
    scale_x = alignment.source_image_width_px / layout.page_width
    scale_y = alignment.source_image_height_px / layout.page_height
    answer_aligned_point = np.array(
        [
            center_x * scale_x,
            alignment.source_image_height_px - (center_y * scale_y),
            1.0,
        ]
    )
    source_point = alignment.answer_aligned_to_source_transform @ answer_aligned_point
    return source_point[0] / source_point[2], source_point[1] / source_point[2]


def _source_raster_point_to_pdf_point(
    *,
    alignment: _AlignedSheet,
    page_width: float,
    page_height: float,
    center_x: float,
    center_y: float,
) -> tuple[float, float]:
    return (
        center_x * page_width / alignment.source_image_width_px,
        page_height - (center_y * page_height / alignment.source_image_height_px),
    )


def _refine_source_answer_option_center(
    *,
    layout: PageLayout,
    alignment: _AlignedSheet,
    column_index: int,
    row_index: int,
    option_index: int,
    center_x: float,
    center_y: float,
) -> tuple[float, float]:
    image = alignment.source_image
    radius_px = _source_bubble_radius(layout, alignment)
    expected_centers: list[tuple[float, float]] = []
    for index in range(len(OPTION_LABELS)):
        option_center_x, option_center_y = layout.answer_option_center(column_index, row_index, index)
        expected_centers.append(
            _layout_point_to_source_raster_point(
                layout=layout,
                alignment=alignment,
                center_x=option_center_x,
                center_y=option_center_y,
            )
        )
    x_values = [point[0] for point in expected_centers]
    y_values = [point[1] for point in expected_centers]
    x0 = max(0, int(round(min(x_values) - radius_px * ANSWER_ROW_REFINEMENT_X_PADDING_RATIO)))
    x1 = min(image.shape[1], int(round(max(x_values) + radius_px * ANSWER_ROW_REFINEMENT_X_PADDING_RATIO)) + 1)
    y0 = max(0, int(round(min(y_values) - radius_px * ANSWER_ROW_REFINEMENT_Y_PADDING_RATIO)))
    y1 = min(image.shape[0], int(round(max(y_values) + radius_px * ANSWER_ROW_REFINEMENT_Y_PADDING_RATIO)) + 1)
    if x1 <= x0 or y1 <= y0:
        return _refine_source_bubble_center(layout=layout, alignment=alignment, center_x=center_x, center_y=center_y)

    circles = _detect_source_bubble_circles(image[y0:y1, x0:x1].copy(), radius_px)
    candidates = [(x0 + circle_x, y0 + circle_y, circle_radius) for circle_x, circle_y, circle_radius in circles]
    row_candidates = _answer_row_bubble_candidates(candidates, radius_px)
    if len(row_candidates) > option_index:
        candidate_x, candidate_y, _ = sorted(row_candidates, key=lambda candidate: candidate[0])[option_index]
        return candidate_x, candidate_y
    if row_candidates:
        candidate_x, candidate_y, _ = min(
            row_candidates,
            key=lambda candidate: abs(candidate[0] - center_x) + (abs(candidate[1] - center_y) * 0.25),
        )
        return candidate_x, candidate_y
    return _refine_source_bubble_center(layout=layout, alignment=alignment, center_x=center_x, center_y=center_y)


def _answer_row_bubble_candidates(
    candidates: list[tuple[float, float, float]],
    radius_px: float,
) -> list[tuple[float, float, float]]:
    if not candidates:
        return []
    y_values = [candidate[1] for candidate in candidates]
    y_span = max(y_values) - min(y_values)
    row_candidates = candidates
    if y_span >= radius_px * ANSWER_ROW_BUBBLE_CLUSTER_MIN_SPAN_RATIO:
        split_y = min(y_values) + (y_span * ANSWER_ROW_BUBBLE_CLUSTER_SPLIT_RATIO)
        lower_candidates = [candidate for candidate in candidates if candidate[1] >= split_y]
        if len(lower_candidates) >= 2:
            row_candidates = lower_candidates

    merged: list[tuple[float, float, float]] = []
    for candidate in sorted(row_candidates, key=lambda item: (item[0], item[1])):
        if merged and abs(candidate[0] - merged[-1][0]) <= radius_px * ANSWER_ROW_CANDIDATE_MERGE_RATIO:
            previous = merged[-1]
            if candidate[2] > previous[2]:
                merged[-1] = candidate
            continue
        merged.append(candidate)
    return merged


def _refine_source_bubble_center(
    *,
    layout: PageLayout,
    alignment: _AlignedSheet,
    center_x: float,
    center_y: float,
) -> tuple[float, float]:
    image = alignment.source_image
    radius_px = _source_bubble_radius(layout, alignment)
    pad = max(24, int(round(radius_px * 2.8)))
    x0 = max(0, int(round(center_x)) - pad)
    x1 = min(image.shape[1], int(round(center_x)) + pad + 1)
    y0 = max(0, int(round(center_y)) - pad)
    y1 = min(image.shape[0], int(round(center_y)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return center_x, center_y

    circles = _detect_source_bubble_circles(image[y0:y1, x0:x1].copy(), radius_px)
    if not circles:
        return center_x, center_y

    candidates: list[tuple[float, float, float]] = []
    for circle_x, circle_y, circle_radius in circles:
        source_x = x0 + float(circle_x)
        source_y = y0 + float(circle_y)
        distance = float(np.hypot(source_x - center_x, source_y - center_y))
        radius_penalty = abs(float(circle_radius) - radius_px) * 0.35
        candidates.append((distance + radius_penalty, source_x, source_y))
    if not candidates:
        return center_x, center_y

    _, best_x, best_y = min(candidates, key=lambda candidate: candidate[0])
    jitter_tolerance = max(HOUGH_REFINEMENT_MIN_JITTER_PX, radius_px * HOUGH_REFINEMENT_JITTER_RATIO)
    if np.hypot(best_x - center_x, best_y - center_y) <= jitter_tolerance:
        return center_x, center_y

    return best_x, best_y


def _detect_source_bubble_circles(roi: np.ndarray, radius_px: float) -> list[tuple[float, float, float]]:
    red_mask = _red_overlay_mask(roi)
    roi[red_mask] = 255
    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(grayscale, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(12.0, radius_px * 1.4),
        param1=80,
        param2=14,
        minRadius=max(3, int(round(radius_px * 0.6))),
        maxRadius=max(4, int(round(radius_px * 1.45))),
    )
    if circles is None:
        return []
    return [(float(circle_x), float(circle_y), float(circle_radius)) for circle_x, circle_y, circle_radius in circles[0]]


def _red_overlay_mask(image: np.ndarray) -> np.ndarray:
    blue = image[:, :, 0].astype(int)
    green = image[:, :, 1].astype(int)
    red = image[:, :, 2].astype(int)
    return (red > green + 15) & (red > blue + 15)


def _source_bubble_radius(layout: PageLayout, alignment: _AlignedSheet) -> float:
    scale_x = alignment.source_image_width_px / layout.page_width
    scale_y = alignment.source_image_height_px / layout.page_height
    return layout.bubble_radius * ((scale_x + scale_y) / 2)


def _overlay_bubble_radius(layout: PageLayout, page_width: float, page_height: float) -> float:
    scale_x = page_width / layout.page_width
    scale_y = page_height / layout.page_height
    return layout.bubble_radius * ((scale_x + scale_y) / 2)


def _compact_json(value: dict | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _wrap_annotation_lines(
    pdf: canvas.Canvas,
    lines: list[str],
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(_wrap_text_to_width(pdf, line, font_name, font_size, max_width))
    return wrapped


def _wrap_text_to_width(
    pdf: canvas.Canvas,
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    lines: list[str] = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if pdf.stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
            continue

        lines.extend(_split_long_token(pdf, current, font_name, font_size, max_width))
        current = word

    lines.extend(_split_long_token(pdf, current, font_name, font_size, max_width))
    return lines


def _split_long_token(
    pdf: canvas.Canvas,
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    if pdf.stringWidth(text, font_name, font_size) <= max_width:
        return [text]

    parts: list[str] = []
    current = ""
    for character in text:
        candidate = f"{current}{character}"
        if current and pdf.stringWidth(candidate, font_name, font_size) > max_width:
            parts.append(current)
            current = character
            continue
        current = candidate

    if current:
        parts.append(current)
    return parts


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
