from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import OPTION_LABELS, STUDENT_ID_COLUMNS, STUDENT_ID_ROWS, PageLayout

OPTION_OUTLINE_THRESHOLD = 0.10
MIN_MARK_SCORE = 0.10
RELATIVE_MARK_THRESHOLD = 0.35
OUTLINE_LIFT_WEIGHT = 0.8
ROW_PRESENCE_THRESHOLD = 0.05


class UnsupportedSheetError(RuntimeError):
    pass


@dataclass(slots=True)
class GradeResult:
    qr_data: dict | str | None
    student_id: str
    marked_answers: dict[str, list[str]]
    omr_error: str = ""


@dataclass(slots=True)
class BatchGradeResult:
    source_pdf: str
    qr_data: dict | str | None
    student_id: str
    marked_answers: dict[str, list[str]]
    omr_error: str


def grade_pdf(
    pdf_path: str | Path,
    layout: PageLayout | None = None,
) -> GradeResult:
    layout = layout or PageLayout()
    pdf_path = Path(pdf_path)
    image = _rasterize_pdf_page(pdf_path)
    try:
        page_aligned_image, answer_aligned_image, qr_data = _align_image_to_layout(image, layout)
    except UnsupportedSheetError as exc:
        raise UnsupportedSheetError(f"{pdf_path}: {exc}") from exc
    page_binary = _threshold_image(page_aligned_image)
    answer_binary = _threshold_image(answer_aligned_image)
    student_id = _grade_student_id(page_binary, layout)
    marked_answers = _grade_answers(answer_binary, layout)
    return GradeResult(
        qr_data=qr_data,
        student_id=student_id,
        marked_answers=marked_answers,
        omr_error="",
    )


def grade_path(
    path: str | Path,
    layout: PageLayout | None = None,
) -> GradeResult | list[BatchGradeResult]:
    target = Path(path)
    if target.is_dir():
        return grade_directory(target, layout=layout)
    return grade_pdf(target, layout=layout)


def grade_directory(
    directory: str | Path,
    layout: PageLayout | None = None,
) -> list[BatchGradeResult]:
    target_dir = Path(directory)
    pdf_paths = sorted(path for path in target_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    results: list[BatchGradeResult] = []
    for pdf_path in pdf_paths:
        try:
            result = grade_pdf(pdf_path, layout=layout)
            results.append(
                BatchGradeResult(
                    source_pdf=pdf_path.name,
                    qr_data=result.qr_data,
                    student_id=result.student_id,
                    marked_answers=result.marked_answers,
                    omr_error="",
                )
            )
        except Exception as exc:
            results.append(
                BatchGradeResult(
                    source_pdf=pdf_path.name,
                    qr_data=None,
                    student_id="",
                    marked_answers={},
                    omr_error=str(exc),
                )
            )
    return results


def _rasterize_pdf_page(pdf_path: Path, dpi: int = 200) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="omr-grade-") as temp_dir:
        output_prefix = Path(temp_dir) / "page"
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                str(dpi),
                "-f",
                "1",
                "-singlefile",
                str(pdf_path),
                str(output_prefix),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        image = cv2.imread(str(output_prefix.with_suffix(".png")), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to rasterize PDF: {pdf_path}")
        return image


def _align_image_to_layout(
    image: np.ndarray, layout: PageLayout
) -> tuple[np.ndarray, np.ndarray, dict | str | None]:
    page_binary = _threshold_image(image)
    page_marker_centers = _detect_page_marker_centers(page_binary, layout)
    expected_page_centers = _expected_marker_centers_px(image, layout, layout.corner_marker_centers())
    page_transform = cv2.getPerspectiveTransform(page_marker_centers, expected_page_centers)
    page_aligned_image = cv2.warpPerspective(
        image,
        page_transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    qr_data = _decode_qr_data(page_aligned_image)
    if qr_data is None:
        qr_data = _decode_qr_data(image)

    answer_binary = _threshold_image(page_aligned_image)
    answer_marker_centers = _detect_answer_marker_centers(answer_binary, layout)
    expected_answer_centers = _expected_marker_centers_px(page_aligned_image, layout, layout.local_marker_centers())
    answer_transform = _similarity_transform_from_two_points(answer_marker_centers, expected_answer_centers)
    answer_aligned_image = cv2.warpAffine(
        page_aligned_image,
        answer_transform,
        (page_aligned_image.shape[1], page_aligned_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return page_aligned_image, answer_aligned_image, qr_data


def _decode_qr_data(image: np.ndarray) -> dict | str | None:
    detector = cv2.QRCodeDetector()
    decoded_text, _, _ = detector.detectAndDecode(image)
    if not decoded_text:
        return None
    try:
        return json.loads(decoded_text)
    except json.JSONDecodeError:
        return decoded_text


def _threshold_image(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresholded


def _detect_page_marker_centers(binary: np.ndarray, layout: PageLayout) -> np.ndarray:
    return _detect_marker_centers(
        binary=binary,
        layout=layout,
        marker_centers=layout.corner_marker_centers(),
        marker_size_pt=layout.corner_marker_size,
        roi_half_size_px=110,
    )


def _detect_answer_marker_centers(binary: np.ndarray, layout: PageLayout) -> np.ndarray:
    return _detect_marker_centers(
        binary=binary,
        layout=layout,
        marker_centers=layout.local_marker_centers(),
        marker_size_pt=layout.local_marker_size,
        roi_half_size_px=80,
    )


def _detect_marker_centers(
    *,
    binary: np.ndarray,
    layout: PageLayout,
    marker_centers: dict[str, tuple[float, float]],
    marker_size_pt: float,
    roi_half_size_px: int,
) -> np.ndarray:
    detected: list[tuple[float, float]] = []
    expected = _expected_marker_centers_px(binary, layout, marker_centers)
    scale = (binary.shape[1] / layout.page_width + binary.shape[0] / layout.page_height) / 2
    min_area = max(20.0, (marker_size_pt * scale) ** 2 * 0.25)

    for name, center in zip(marker_centers.keys(), expected, strict=True):
        cx, cy = center
        roi, x0, y0 = _extract_roi(binary, cx, cy, roi_half_size_px)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = float("-inf")
        best_center: tuple[float, float] | None = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width == 0 or height == 0:
                continue

            aspect_ratio = width / height
            if not 0.65 <= aspect_ratio <= 1.35:
                continue

            fill_ratio = area / float(width * height)
            if fill_ratio < 0.45:
                continue

            center_x = x0 + x + (width / 2.0)
            center_y = y0 + y + (height / 2.0)
            distance_penalty = np.hypot(center_x - cx, center_y - cy) * 2.0
            score = area - distance_penalty
            if score > best_score:
                best_score = score
                best_center = (center_x, center_y)

        if best_center is None:
            raise UnsupportedSheetError(f"Required alignment marker '{name}' was not detected")

        detected.append(best_center)

    return np.array(detected, dtype=np.float32)


def _extract_roi(binary: np.ndarray, center_x: float, center_y: float, half_size: int) -> tuple[np.ndarray, int, int]:
    x0 = max(0, int(round(center_x)) - half_size)
    x1 = min(binary.shape[1], int(round(center_x)) + half_size)
    y0 = max(0, int(round(center_y)) - half_size)
    y1 = min(binary.shape[0], int(round(center_y)) + half_size)
    return binary[y0:y1, x0:x1], x0, y0


def _expected_marker_centers_px(
    image_or_binary: np.ndarray,
    layout: PageLayout,
    marker_centers: dict[str, tuple[float, float]],
) -> np.ndarray:
    page_height_px, page_width_px = image_or_binary.shape[:2]
    scale_x = page_width_px / layout.page_width
    scale_y = page_height_px / layout.page_height
    points: list[tuple[float, float]] = []
    for center_x_pt, center_y_pt in marker_centers.values():
        points.append((center_x_pt * scale_x, page_height_px - (center_y_pt * scale_y)))
    return np.array(points, dtype=np.float32)


def _similarity_transform_from_two_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_p1, src_p2 = src
    dst_p1, dst_p2 = dst
    src_vector = src_p2 - src_p1
    dst_vector = dst_p2 - dst_p1
    src_perp = np.array([-src_vector[1], src_vector[0]], dtype=np.float32)
    dst_perp = np.array([-dst_vector[1], dst_vector[0]], dtype=np.float32)
    src_points = np.array([src_p1, src_p2, src_p1 + src_perp], dtype=np.float32)
    dst_points = np.array([dst_p1, dst_p2, dst_p1 + dst_perp], dtype=np.float32)
    return cv2.getAffineTransform(src_points, dst_points)


def _grade_student_id(binary: np.ndarray, layout: PageLayout) -> str:
    digits: list[str] = []
    for column_index in range(STUDENT_ID_COLUMNS):
        column_scores: list[float] = []
        for row_index in range(STUDENT_ID_ROWS):
            center_x_pt, center_y_pt = layout.student_id_bubble_center(column_index, row_index)
            column_scores.append(_fill_score(binary, layout, center_x_pt, center_y_pt))
        digits.append(str(int(np.argmax(column_scores))))
    return "".join(digits)


def _grade_answers(binary: np.ndarray, layout: PageLayout) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    question_number = 1

    for column_index in range(layout.answer_columns_per_page):
        for row_index in range(layout.questions_per_column):
            row_outline_sum = _answer_row_outline_sum(binary, layout, column_index, row_index)
            if row_outline_sum <= ROW_PRESENCE_THRESHOLD:
                continue

            fill_scores: list[float] = []
            outline_scores: list[float] = []
            option_presence: list[bool] = []
            for option_index in range(len(OPTION_LABELS)):
                center_x_pt, center_y_pt = layout.answer_option_center(column_index, row_index, option_index)
                outline_score = _outline_score(binary, layout, center_x_pt, center_y_pt)
                outline_scores.append(outline_score)
                fill_scores.append(_fill_score(binary, layout, center_x_pt, center_y_pt))
                option_presence.append(outline_score > OPTION_OUTLINE_THRESHOLD)

            option_count = _infer_option_count(option_presence)
            if option_count == 0:
                continue

            marked = _marked_option_indexes(
                fill_scores[:option_count],
                option_presence[:option_count],
                outline_scores[:option_count],
            )
            answers[str(question_number)] = [OPTION_LABELS[index] for index in marked]
            question_number += 1

    return answers


def _answer_row_outline_sum(binary: np.ndarray, layout: PageLayout, column_index: int, row_index: int) -> float:
    total = 0.0
    for option_index in range(len(OPTION_LABELS)):
        center_x_pt, center_y_pt = layout.answer_option_center(column_index, row_index, option_index)
        total += _outline_score(binary, layout, center_x_pt, center_y_pt)
    return total


def _infer_option_count(option_presence: list[bool]) -> int:
    present_indexes = [index for index, present in enumerate(option_presence) if present]
    if len(present_indexes) < 2:
        return 0
    return present_indexes[-1] + 1


def _marked_option_indexes(
    fill_scores: list[float],
    option_presence: list[bool],
    outline_scores: list[float],
) -> list[int]:
    if not fill_scores:
        return []

    baseline_outline = float(np.median(outline_scores))
    mark_scores: list[float] = []
    for fill_score, outline_score, present in zip(fill_scores, outline_scores, option_presence, strict=True):
        if not present:
            mark_scores.append(0.0)
            continue
        outline_lift = max(0.0, outline_score - baseline_outline)
        mark_scores.append(fill_score + (OUTLINE_LIFT_WEIGHT * outline_lift))

    max_score = max(mark_scores)
    if max_score < MIN_MARK_SCORE:
        return []

    threshold = max(MIN_MARK_SCORE, max_score * RELATIVE_MARK_THRESHOLD)
    return [index for index, score in enumerate(mark_scores) if score >= threshold]


def _outline_score(binary: np.ndarray, layout: PageLayout, center_x_pt: float, center_y_pt: float) -> float:
    center_x_px, center_y_px, radius_px = _bubble_geometry_px(binary, layout, center_x_pt, center_y_pt)
    local_patch, local_x, local_y = _bubble_patch(binary, center_x_px, center_y_px, radius_px, scale=1.4)
    distance = np.sqrt((local_x - center_x_px) ** 2 + (local_y - center_y_px) ** 2)
    ring_mask = (distance >= radius_px * 0.75) & (distance <= radius_px * 1.25)
    if not np.any(ring_mask):
        return 0.0
    return float(local_patch[ring_mask].mean() / 255.0)


def _fill_score(binary: np.ndarray, layout: PageLayout, center_x_pt: float, center_y_pt: float) -> float:
    center_x_px, center_y_px, radius_px = _bubble_geometry_px(binary, layout, center_x_pt, center_y_pt)
    local_patch, local_x, local_y = _bubble_patch(binary, center_x_px, center_y_px, radius_px, scale=0.7)
    distance = np.sqrt((local_x - center_x_px) ** 2 + (local_y - center_y_px) ** 2)
    fill_mask = distance <= radius_px * 0.58
    if not np.any(fill_mask):
        return 0.0
    return float(local_patch[fill_mask].mean() / 255.0)


def _bubble_geometry_px(
    binary: np.ndarray,
    layout: PageLayout,
    center_x_pt: float,
    center_y_pt: float,
) -> tuple[float, float, float]:
    scale_x = binary.shape[1] / layout.page_width
    scale_y = binary.shape[0] / layout.page_height
    center_x_px = center_x_pt * scale_x
    center_y_px = binary.shape[0] - (center_y_pt * scale_y)
    radius_px = layout.bubble_radius * (scale_x + scale_y) / 2
    return center_x_px, center_y_px, radius_px


def _bubble_patch(
    binary: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    radius_px: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pad = max(2, int(np.ceil(radius_px * scale)))
    x0 = max(0, int(np.floor(center_x_px - pad)))
    x1 = min(binary.shape[1], int(np.ceil(center_x_px + pad + 1)))
    y0 = max(0, int(np.floor(center_y_px - pad)))
    y1 = min(binary.shape[0], int(np.ceil(center_y_px + pad + 1)))
    patch = binary[y0:y1, x0:x1]
    y_indices, x_indices = np.ogrid[y0:y1, x0:x1]
    return patch, x_indices, y_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grade a filled OMR sheet PDF or a folder of PDFs.")
    parser.add_argument("path", help="Path to a filled OMR sheet PDF or a directory containing PDF files.")
    parser.add_argument(
        "--output",
        help="Optional annotated PDF output path. For folder input, this must be a directory path.",
    )
    parser.add_argument(
        "--correct-answers",
        help="Optional answer key as inline JSON or a path to a JSON file. Correct answers are watermarked in faint red.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        correct_answers = _load_correct_answers(args.correct_answers)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = grade_path(args.path, output_path=args.output, correct_answers=correct_answers)
    except UnsupportedSheetError as exc:
        parser.exit(2, f"{exc}\n")
    if isinstance(result, list):
        payload = [asdict(item) for item in result]
    else:
        payload = asdict(result)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
