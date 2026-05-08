from __future__ import annotations

import argparse
import json
from math import ceil, floor
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import OPTION_LABELS, STUDENT_ID_COLUMNS, STUDENT_ID_ROWS, PageLayout

OPTION_OUTLINE_THRESHOLD = 0.06
OPTION_TRAILING_OUTLINE_THRESHOLD = 0.09
MIN_MARK_SCORE = 0.10
RELATIVE_MARK_THRESHOLD = 0.60
OPTION_MARK_MIN_SCORE = 0.45
OPTION_WEAK_MARK_MIN_FILL_SCORE = 0.50
OPTION_WEAK_MARK_MIN_OUTLINE_SCORE = 0.30
OPTION_MULTI_MARK_MIN_TOP_SCORE = 0.75
OUTLINE_LIFT_WEIGHT = 0.8
ROW_PRESENCE_THRESHOLD = 0.05
STUDENT_ID_MIN_SCORE = 0.25
STUDENT_ID_DOMINANCE_RATIO = 1.8
STUDENT_ID_MIN_SCORE_MARGIN = 0.18
STUDENT_ID_ALIGNMENT_SEARCH_RADIUS_PT = 24.0
STUDENT_ID_ALIGNMENT_COARSE_STEP_PT = 2.0
STUDENT_ID_ALIGNMENT_FINE_RADIUS_PT = 3.0
STUDENT_ID_ALIGNMENT_FINE_STEP_PT = 1.0
STUDENT_ID_ALIGNMENT_OFFSET_PENALTY = 0.03
QR_CROP_PADDING_PT = 8.0
QR_CROP_SCALE_FACTORS = (1, 2, 3, 4)
TOP_RIGHT_MARKER_MAX_DOWNWARD_DRIFT_RATIO = 2.5


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


@dataclass(frozen=True, slots=True)
class _AlignedSheet:
    source_image: np.ndarray
    page_aligned_image: np.ndarray
    answer_aligned_image: np.ndarray
    qr_data: dict | str | None
    answer_aligned_to_source_transform: np.ndarray
    source_image_width_px: int
    source_image_height_px: int


def grade_pdf(
    pdf_path: str | Path,
    layout: PageLayout | None = None,
) -> GradeResult:
    result, _ = _grade_pdf_with_alignment(pdf_path, layout=layout)
    return result


def _grade_pdf_with_alignment(
    pdf_path: str | Path,
    layout: PageLayout | None = None,
) -> tuple[GradeResult, _AlignedSheet]:
    layout = layout or PageLayout()
    pdf_path = Path(pdf_path)
    image = _rasterize_pdf_page(pdf_path)
    try:
        aligned_sheet = _align_image_to_layout(image, layout)
    except UnsupportedSheetError as exc:
        raise UnsupportedSheetError(f"{pdf_path}: {exc}") from exc

    page_binary = _threshold_image(aligned_sheet.page_aligned_image)
    answer_binary = _threshold_image(aligned_sheet.answer_aligned_image)
    marked_answers = _grade_answers(answer_binary, layout)

    student_id = ""
    student_id_error = ""
    try:
        student_id = _grade_student_id_with_local_alignment(page_binary, layout)
    except UnsupportedSheetError as exc:
        student_id_error = str(exc)

    return (
        GradeResult(
            qr_data=aligned_sheet.qr_data,
            student_id=student_id,
            marked_answers=marked_answers,
            omr_error=student_id_error,
        ),
        aligned_sheet,
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
                    omr_error=f"{pdf_path}: {result.omr_error}" if result.omr_error else "",
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


def _align_image_to_layout(image: np.ndarray, layout: PageLayout) -> _AlignedSheet:
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
        qr_data = _decode_qr_data_from_layout(page_aligned_image, layout)
    if qr_data is None:
        qr_data = _decode_qr_data(image)

    answer_binary = _threshold_image(page_aligned_image)
    answer_marker_centers = _detect_answer_marker_centers(answer_binary, layout)
    expected_answer_centers = _expected_marker_centers_px(page_aligned_image, layout, layout.local_marker_centers())
    answer_transform = _similarity_transform_from_two_points(answer_marker_centers, expected_answer_centers)
    source_to_answer_transform = _affine_to_homography(answer_transform) @ page_transform
    answer_aligned_image = cv2.warpAffine(
        page_aligned_image,
        answer_transform,
        (page_aligned_image.shape[1], page_aligned_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return _AlignedSheet(
        source_image=image,
        page_aligned_image=page_aligned_image,
        answer_aligned_image=answer_aligned_image,
        qr_data=qr_data,
        answer_aligned_to_source_transform=np.linalg.inv(source_to_answer_transform),
        source_image_width_px=image.shape[1],
        source_image_height_px=image.shape[0],
    )


def _affine_to_homography(transform: np.ndarray) -> np.ndarray:
    return np.vstack([transform, np.array([0.0, 0.0, 1.0])])


def _decode_qr_data(image: np.ndarray) -> dict | str | None:
    detector = cv2.QRCodeDetector()
    decoded_text, _, _ = detector.detectAndDecode(image)
    return _parse_qr_data(decoded_text)


def _decode_qr_data_from_layout(image: np.ndarray, layout: PageLayout) -> dict | str | None:
    qr_crop = _crop_qr_region(image, layout)
    if qr_crop is None:
        return None

    detector = cv2.QRCodeDetector()
    for scale_factor in QR_CROP_SCALE_FACTORS:
        candidate = qr_crop
        if scale_factor != 1:
            candidate = cv2.resize(
                qr_crop,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
        decoded_text, _, _ = detector.detectAndDecode(candidate)
        decoded_data = _parse_qr_data(decoded_text)
        if decoded_data is not None:
            return decoded_data
    return None


def _crop_qr_region(image: np.ndarray, layout: PageLayout) -> np.ndarray | None:
    height_px, width_px = image.shape[:2]
    scale_x = width_px / layout.page_width
    scale_y = height_px / layout.page_height

    left_pt = max(0.0, layout.qr_box_left - QR_CROP_PADDING_PT)
    right_pt = min(layout.page_width, layout.qr_box_left + layout.qr_size + QR_CROP_PADDING_PT)
    bottom_pt = max(0.0, layout.qr_box_bottom - QR_CROP_PADDING_PT)
    top_pt = min(layout.page_height, layout.qr_box_bottom + layout.qr_size + QR_CROP_PADDING_PT)

    x0 = max(0, floor(left_pt * scale_x))
    x1 = min(width_px, ceil(right_pt * scale_x))
    y0 = max(0, floor(height_px - (top_pt * scale_y)))
    y1 = min(height_px, ceil(height_px - (bottom_pt * scale_y)))

    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _parse_qr_data(decoded_text: str) -> dict | str | None:
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
        roi_half_size_px=200,
    )


def _detect_answer_marker_centers(binary: np.ndarray, layout: PageLayout) -> np.ndarray:
    return _detect_marker_centers(
        binary=binary,
        layout=layout,
        marker_centers=layout.local_marker_centers(),
        marker_size_pt=layout.local_marker_size,
        roi_half_size_px=120,
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
    expected_size = marker_size_pt * scale
    expected_area = expected_size**2
    min_size = expected_size * 0.45
    max_size = expected_size * 1.80
    min_area = max(20.0, expected_area * 0.18)

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
            if width < min_size or height < min_size or width > max_size or height > max_size:
                continue

            aspect_ratio = width / height
            if not 0.65 <= aspect_ratio <= 1.35:
                continue

            fill_ratio = area / float(width * height)
            if fill_ratio < 0.25:
                continue

            center_x = x0 + x + (width / 2.0)
            center_y = y0 + y + (height / 2.0)
            distance_ratio = np.hypot(center_x - cx, center_y - cy) / expected_size
            width_ratio = width / expected_size
            height_ratio = height / expected_size
            area_ratio = area / expected_area
            size_penalty = abs(width_ratio - 1.0) + abs(height_ratio - 1.0)
            score = area_ratio + fill_ratio - (1.10 * distance_ratio) - (0.85 * size_penalty)
            if score > best_score:
                best_score = score
                best_center = (center_x, center_y)

        if best_center is None:
            raise UnsupportedSheetError(f"Required alignment marker '{name}' was not detected")
        if (
            name == "top_right"
            and best_center[1] - cy > expected_size * TOP_RIGHT_MARKER_MAX_DOWNWARD_DRIFT_RATIO
        ):
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


def _grade_student_id_with_local_alignment(binary: np.ndarray, layout: PageLayout) -> str:
    try:
        return _grade_student_id(binary, layout)
    except UnsupportedSheetError:
        offset_x, offset_y = _find_student_id_grid_offset(binary, layout)
        last_error: UnsupportedSheetError | None = None
        for candidate_x, candidate_y in _nearby_student_id_offsets(offset_x, offset_y):
            try:
                return _grade_student_id(binary, layout, offset_x=candidate_x, offset_y=candidate_y)
            except UnsupportedSheetError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return _grade_student_id(binary, layout, offset_x=offset_x, offset_y=offset_y)


def _grade_student_id(
    binary: np.ndarray,
    layout: PageLayout,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> str:
    digits: list[str] = []
    for column_index in range(STUDENT_ID_COLUMNS):
        column_scores: list[float] = []
        for row_index in range(STUDENT_ID_ROWS):
            center_x_pt, center_y_pt = layout.student_id_bubble_center(column_index, row_index)
            column_scores.append(_fill_score(binary, layout, center_x_pt + offset_x, center_y_pt + offset_y))
        max_score = max(column_scores)
        if max_score < STUDENT_ID_MIN_SCORE:
            raise UnsupportedSheetError(f"Student ID column {column_index + 1} is empty")
        sorted_scores = sorted(column_scores, reverse=True)
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        dominance = max_score / second_score if second_score > 0 else float("inf")
        if dominance < STUDENT_ID_DOMINANCE_RATIO and max_score - second_score < STUDENT_ID_MIN_SCORE_MARGIN:
            raise UnsupportedSheetError(f"Student ID column {column_index + 1} has multiple marks")
        digits.append(str(column_scores.index(max_score)))
    return "".join(digits)


def _find_student_id_grid_offset(binary: np.ndarray, layout: PageLayout) -> tuple[float, float]:
    best_score = float("-inf")
    best_offset = (0.0, 0.0)

    for offset_x, offset_y in _iter_offset_grid(
        -STUDENT_ID_ALIGNMENT_SEARCH_RADIUS_PT,
        STUDENT_ID_ALIGNMENT_SEARCH_RADIUS_PT,
        STUDENT_ID_ALIGNMENT_COARSE_STEP_PT,
    ):
        score = _student_id_grid_outline_score(binary, layout, offset_x, offset_y)
        if score > best_score:
            best_score = score
            best_offset = (offset_x, offset_y)

    coarse_x, coarse_y = best_offset
    fine_min_x = coarse_x - STUDENT_ID_ALIGNMENT_FINE_RADIUS_PT
    fine_max_x = coarse_x + STUDENT_ID_ALIGNMENT_FINE_RADIUS_PT
    fine_min_y = coarse_y - STUDENT_ID_ALIGNMENT_FINE_RADIUS_PT
    fine_max_y = coarse_y + STUDENT_ID_ALIGNMENT_FINE_RADIUS_PT
    for offset_x in _iter_float_range(fine_min_x, fine_max_x, STUDENT_ID_ALIGNMENT_FINE_STEP_PT):
        for offset_y in _iter_float_range(fine_min_y, fine_max_y, STUDENT_ID_ALIGNMENT_FINE_STEP_PT):
            score = _student_id_grid_outline_score(binary, layout, offset_x, offset_y)
            if score > best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

    return best_offset


def _nearby_student_id_offsets(offset_x: float, offset_y: float):
    candidates: list[tuple[float, float]] = []
    for delta_x in (-1.0, 0.0, 1.0):
        for delta_y in (-1.0, 0.0, 1.0):
            candidates.append((offset_x + delta_x, offset_y + delta_y))
    return sorted(candidates, key=lambda point: np.hypot(point[0] - offset_x, point[1] - offset_y))


def _student_id_grid_outline_score(binary: np.ndarray, layout: PageLayout, offset_x: float, offset_y: float) -> float:
    total = 0.0
    for column_index in range(STUDENT_ID_COLUMNS):
        for row_index in range(STUDENT_ID_ROWS):
            center_x_pt, center_y_pt = layout.student_id_bubble_center(column_index, row_index)
            total += _outline_score(binary, layout, center_x_pt + offset_x, center_y_pt + offset_y)
    return total - (STUDENT_ID_ALIGNMENT_OFFSET_PENALTY * np.hypot(offset_x, offset_y))


def _iter_offset_grid(minimum: float, maximum: float, step: float):
    for offset_x in _iter_float_range(minimum, maximum, step):
        for offset_y in _iter_float_range(minimum, maximum, step):
            yield offset_x, offset_y


def _iter_float_range(minimum: float, maximum: float, step: float):
    value = minimum
    while value <= maximum + (step / 2.0):
        yield float(value)
        value += step


def _marked_student_digit_indexes(fill_scores: list[float]) -> list[int]:
    if not fill_scores:
        return []

    max_score = max(fill_scores)
    if max_score < MIN_MARK_SCORE:
        return []

    threshold = max(MIN_MARK_SCORE, max_score * RELATIVE_MARK_THRESHOLD)
    return [index for index, score in enumerate(fill_scores) if score >= threshold]


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

            option_count = _infer_option_count(outline_scores)
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


def _infer_option_count(outline_scores: list[float]) -> int:
    present_indexes = [index for index, score in enumerate(outline_scores) if score > OPTION_OUTLINE_THRESHOLD]
    if len(present_indexes) < 2:
        return 0
    option_count = present_indexes[-1] + 1
    while option_count > 2 and outline_scores[option_count - 1] < OPTION_TRAILING_OUTLINE_THRESHOLD:
        option_count -= 1
    return option_count


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
    if max_score < OPTION_MARK_MIN_SCORE:
        return []

    threshold = max(OPTION_MARK_MIN_SCORE, max_score * RELATIVE_MARK_THRESHOLD)
    marked = []
    for index, score in enumerate(mark_scores):
        if score >= threshold:
            marked.append(index)
        elif (
            score >= OPTION_MARK_MIN_SCORE
            and fill_scores[index] >= OPTION_WEAK_MARK_MIN_FILL_SCORE
            and outline_scores[index] >= OPTION_WEAK_MARK_MIN_OUTLINE_SCORE
        ):
            marked.append(index)
    if len(marked) > 1 and max_score < OPTION_MULTI_MARK_MIN_TOP_SCORE:
        return [int(np.argmax(mark_scores))]
    return marked


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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = grade_path(args.path)
    except UnsupportedSheetError as exc:
        parser.exit(2, f"{exc}\n")
    if isinstance(result, list):
        payload = [asdict(item) for item in result]
    else:
        payload = asdict(result)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
