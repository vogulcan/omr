from __future__ import annotations

import json
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import black, white
from reportlab.pdfgen import canvas

import omr.annotate as annotate_module
from omr.cli import parse_choice_count, parse_question_count
from omr.annotate import annotate_directory, annotate_pdf, load_correct_answers, _refine_source_bubble_center
from omr.grade import (
    _AlignedSheet,
    UnsupportedSheetError,
    _decode_qr_data_from_layout,
    _detect_answer_marker_centers,
    _detect_page_marker_centers,
    _rasterize_pdf_page,
    _threshold_image,
    grade_directory,
    grade_path,
    grade_pdf,
)
from omr.generator import DUMMY_QR_DATA, dummy_qr_payload, generate_omr_sheet
from omr.layout import MAX_QUESTIONS_PER_PAGE, OPTION_LABELS, STUDENT_ID_COLUMNS, STUDENT_ID_ROWS, PageLayout, paginate_questions
from omr.models import MAX_QUESTION_COUNT, SheetConfig
from omr.pdf_fonts import get_pdf_fonts

ROOT = Path(__file__).resolve().parents[1]
ANSWER_KEY_JSON = Path(__file__).resolve().parent / "answer-key.json"
TEST_EXAM_SET_ID = "11111111-2222-3333-4444-555555555555"
TEST_VARIANT_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _merge_test_overlay(source_pdf: Path, overlay_pdf_bytes: bytes, target_pdf: Path) -> None:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    overlay_reader = PdfReader(io.BytesIO(overlay_pdf_bytes))
    for page in reader.pages:
        writer.add_page(page)
    writer.pages[0].merge_page(overlay_reader.pages[0])
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _write_shifted_answer_rows_pdf(
    *,
    source_pdf: Path,
    target_pdf: Path,
    layout: PageLayout,
    answers: dict[int, list[str]],
    student_id: str,
    row_count: int,
    shift_y: float,
) -> None:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(layout.page_width, layout.page_height))
    column_x = layout.margin
    erase_left = column_x - 4
    erase_right = column_x + layout.question_block_width + 4
    erase_top = layout.answer_top_y + 18
    erase_bottom = layout.answer_top_y - ((row_count - 1) * layout.answer_row_height) - 18 + shift_y

    pdf.setFillColor(white)
    pdf.setStrokeColor(white)
    pdf.rect(erase_left, erase_bottom, erase_right - erase_left, erase_top - erase_bottom, stroke=0, fill=1)

    pdf.setStrokeColor(black)
    pdf.setFillColor(black)
    for row_index in range(row_count):
        row_y = layout.answer_top_y - row_index * layout.answer_row_height + shift_y
        pdf.setFont("Helvetica-Bold", 8)
        pdf.drawRightString(column_x + layout.answer_label_width, row_y - 1, f"{row_index + 1}.")

        pdf.setFont("Helvetica", 7)
        for option_index in range(5):
            center_x, center_y = layout.answer_option_center(0, row_index, option_index)
            center_y += shift_y
            pdf.circle(center_x, center_y, layout.bubble_radius)
            pdf.drawCentredString(center_x, center_y + 6, OPTION_LABELS[option_index])

    for question_number, labels in answers.items():
        row_index = question_number - 1
        for label in labels:
            option_index = OPTION_LABELS.index(label)
            center_x, center_y = layout.answer_option_center(0, row_index, option_index)
            pdf.circle(center_x, center_y + shift_y, layout.bubble_radius * 0.58, stroke=0, fill=1)

    for column_index, digit in enumerate(student_id):
        center_x, center_y = layout.student_id_bubble_center(column_index, int(digit))
        pdf.circle(center_x, center_y, layout.bubble_radius * 0.58, stroke=0, fill=1)

    pdf.save()
    _merge_test_overlay(source_pdf, buffer.getvalue(), target_pdf)


def test_config_rejects_nonpositive_question_count() -> None:
    with pytest.raises(ValueError, match="question_count must be at least 1"):
        SheetConfig(question_count=0, choice_count=4, exam_set_id=TEST_EXAM_SET_ID, variant_id=TEST_VARIANT_ID)


def test_config_rejects_question_count_over_hundred() -> None:
    with pytest.raises(ValueError, match="question_count must not exceed 100"):
        SheetConfig(
            question_count=MAX_QUESTION_COUNT + 1,
            choice_count=4,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        )


@pytest.mark.parametrize(("exam_set_id", "variant_id"), [("", TEST_VARIANT_ID), (TEST_EXAM_SET_ID, "")])
def test_config_rejects_empty_qr_ids(exam_set_id: str, variant_id: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SheetConfig(question_count=2, choice_count=4, exam_set_id=exam_set_id, variant_id=variant_id)


@pytest.mark.parametrize("invalid_count", [0, 1, 6])
def test_config_rejects_invalid_choice_count(invalid_count: int) -> None:
    with pytest.raises(ValueError, match="choice_count must be between 2 and 5"):
        SheetConfig(
            question_count=3,
            choice_count=invalid_count,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        )


def test_config_exposes_uniform_question_option_counts() -> None:
    config = SheetConfig(
        question_count=4,
        choice_count=5,
        exam_set_id=TEST_EXAM_SET_ID,
        variant_id=TEST_VARIANT_ID,
    )
    assert config.question_option_counts == [5, 5, 5, 5]


def test_student_id_area_dimensions_are_fixed() -> None:
    layout = PageLayout()
    assert STUDENT_ID_COLUMNS == 8
    assert STUDENT_ID_ROWS == 10
    assert layout.student_id_block_width > 0
    assert layout.student_id_block_height > 0


def test_question_pagination_caps_at_twenty_rows_per_column() -> None:
    config = SheetConfig(
        question_count=22,
        choice_count=4,
        exam_set_id=TEST_EXAM_SET_ID,
        variant_id=TEST_VARIANT_ID,
    )
    pages = paginate_questions(config)

    assert len(pages) == 1
    assert pages[0][19].column_index == 0
    assert pages[0][19].row_index == 19
    assert pages[0][20].column_index == 1
    assert pages[0][20].row_index == 0


def test_layout_caps_questions_per_page_at_hundred() -> None:
    layout = PageLayout()

    assert layout.option_spacing - layout.bubble_diameter >= 3.0
    assert layout.answer_columns_per_page == 5
    assert layout.questions_per_column == 20
    assert layout.questions_per_page == MAX_QUESTIONS_PER_PAGE
    last_bubble_x, _ = layout.answer_option_center(layout.answer_columns_per_page - 1, 0, 4)
    assert last_bubble_x + layout.bubble_radius < layout.page_width - layout.margin


def test_question_option_labels_match_choice_count() -> None:
    config = SheetConfig(
        question_count=4,
        choice_count=4,
        exam_set_id=TEST_EXAM_SET_ID,
        variant_id=TEST_VARIANT_ID,
    )
    page = paginate_questions(config)[0]

    for placement in page:
        assert placement.option_count == config.choice_count
        assert list(OPTION_LABELS[: config.choice_count]) == list(OPTION_LABELS[: placement.option_count])


def test_hundred_questions_fit_on_a_single_page() -> None:
    layout = PageLayout()
    config = SheetConfig(
        question_count=layout.questions_per_page,
        choice_count=5,
        exam_set_id=TEST_EXAM_SET_ID,
        variant_id=TEST_VARIANT_ID,
    )
    pages = paginate_questions(config, layout)

    assert len(pages) == 1
    assert len(pages[0]) == layout.questions_per_page
    assert pages[0][-1].question_number == layout.questions_per_page
    assert pages[0][-1].column_index == layout.answer_columns_per_page - 1
    assert pages[0][-1].row_index == layout.questions_per_column - 1


def test_parse_question_count() -> None:
    assert parse_question_count("50") == 50


def test_parse_choice_count() -> None:
    assert parse_choice_count("4") == 4


def test_dummy_qr_payload_is_expected_json() -> None:
    assert dummy_qr_payload(
        SheetConfig(
            question_count=1,
            choice_count=4,
            exam_set_id=DUMMY_QR_DATA["examSetId"],
            variant_id=DUMMY_QR_DATA["variantId"],
        )
    ) == (
        '{"examSetId":"f6adcc63-71dc-412c-9c8d-a4609df454ff",'
        '"variantId":"37e3d65f-e540-4e34-b438-549e731be3b0"}'
    )


def test_generate_single_page_pdf(generated_tmp_dir: Path) -> None:
    target = generated_tmp_dir / "single-page.pdf"
    generate_omr_sheet(
        SheetConfig(
            question_count=20,
            choice_count=4,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        ),
        target,
    )

    data = target.read_bytes()
    assert target.exists()
    assert b"Page 1" not in data
    assert b"Dummy QR code" not in data
    assert len(re.findall(rb"/Type /Page\b", data)) == 1


def test_pdf_fonts_prefer_latin_modern() -> None:
    fonts = get_pdf_fonts()
    assert fonts.regular == "OMRLatinModern-Regular"
    assert fonts.bold == "OMRLatinModern-Bold"


def test_generated_sheet_includes_handwritten_fields(generated_tmp_dir: Path) -> None:
    target = generated_tmp_dir / "handwritten-fields.pdf"
    generate_omr_sheet(
        SheetConfig(
            question_count=10,
            choice_count=4,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        ),
        target,
    )

    text = PdfReader(str(target)).pages[0].extract_text()
    assert "Name" in text
    assert "ID" in text
    assert "Signature" in text


def test_generated_sheet_contains_detectable_markers(generated_tmp_dir: Path) -> None:
    target = generated_tmp_dir / "marker-sheet.pdf"
    layout = PageLayout()
    generate_omr_sheet(
        SheetConfig(
            question_count=6,
            choice_count=4,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        ),
        target,
    )

    image = _rasterize_pdf_page(target)
    binary = _threshold_image(image)

    corner_markers = _detect_page_marker_centers(binary, layout)
    local_markers = _detect_answer_marker_centers(binary, layout)

    assert corner_markers.shape == (4, 2)
    assert local_markers.shape == (2, 2)


def test_layout_qr_crop_recovers_downsampled_page(generated_tmp_dir: Path) -> None:
    target = generated_tmp_dir / "qr-crop-sheet.pdf"
    layout = PageLayout()
    generate_omr_sheet(
        SheetConfig(
            question_count=50,
            choice_count=4,
            exam_set_id=DUMMY_QR_DATA["examSetId"],
            variant_id=DUMMY_QR_DATA["variantId"],
        ),
        target,
    )

    image = _rasterize_pdf_page(target)
    small = cv2.resize(image, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    degraded = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    assert _decode_qr_data_from_layout(degraded, layout) == DUMMY_QR_DATA


def test_marker_geometry_does_not_overlap_layout_regions() -> None:
    layout = PageLayout()
    for center_x, center_y in layout.corner_marker_centers().values():
        assert center_x < layout.margin or center_x > layout.page_width - layout.margin
        assert center_y < layout.page_height - layout.margin or center_y > layout.page_height - layout.student_id_top

    left_marker, right_marker = layout.local_marker_centers().values()
    assert left_marker[1] > layout.answer_top_y + 10
    assert right_marker[1] > layout.answer_top_y + 10
    assert left_marker[0] < layout.answer_option_center(0, 0, 0)[0]
    assert right_marker[0] > layout.answer_option_center(layout.answer_columns_per_page - 1, 0, 4)[0]
    assert layout.handwritten_block_bottom_y > right_marker[1] + layout.local_marker_half_size
    assert layout.annotation_box_top_y < layout.answer_bottom_y
    assert layout.annotation_box_bottom_y > layout.corner_marker_centers()["bottom_right"][1] + layout.corner_marker_half_size


def test_generate_hundred_question_pdf_on_single_page(generated_tmp_dir: Path) -> None:
    layout = PageLayout()
    target = generated_tmp_dir / "hundred-question.pdf"
    generate_omr_sheet(
        SheetConfig(
            question_count=layout.questions_per_page,
            choice_count=5,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        ),
        target,
    )

    data = target.read_bytes()
    assert len(re.findall(rb"/Type /Page\b", data)) == 1
    assert b"Dummy QR code" not in data


def test_generation_cli_writes_pdf(generated_tmp_dir: Path, cli_env: dict[str, str]) -> None:
    output_pdf = generated_tmp_dir / "cli-sheet.pdf"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--questions",
            "4",
            "--choices",
            "5",
            "--exam-set-id",
            TEST_EXAM_SET_ID,
            "--variant-id",
            TEST_VARIANT_ID,
            "--output",
            str(output_pdf),
        ],
        cwd=ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == ""
    assert output_pdf.exists()


def test_grade_markerless_sheet_fails_cleanly(sample_pdfs: dict[str, Path]) -> None:
    with pytest.raises(UnsupportedSheetError, match="alignment marker"):
        grade_pdf(sample_pdfs["markerless"])


def test_grade_answered_sample_pdf(sample_pdfs: dict[str, Path]) -> None:
    result = grade_pdf(sample_pdfs["sample_answered"])

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "63620147"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["D"],
        "2": ["C"],
        "3": ["B", "C"],
        "4": ["C"],
        "5": ["A"],
        "6": ["E"],
    }


def test_grade_answer1_pdf(sample_pdfs: dict[str, Path]) -> None:
    result = grade_pdf(sample_pdfs["answer1"])

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "01345072"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["D"],
        "2": ["C"],
        "3": ["B", "D"],
        "4": ["C"],
        "5": ["B"],
        "6": ["D"],
    }


def test_grade_pdf_rejects_empty_student_id_column(sample_pdfs: dict[str, Path]) -> None:
    with pytest.raises(UnsupportedSheetError, match="Student ID column 8 is empty"):
        grade_pdf(sample_pdfs["missing_student_digit"])


def test_grade_directory_reads_all_pdfs(generated_tmp_dir: Path, sample_pdfs: dict[str, Path]) -> None:
    shutil.copy(sample_pdfs["sample_answered"], generated_tmp_dir / "student-a.pdf")
    shutil.copy(sample_pdfs["sample_answered"], generated_tmp_dir / "student-b.pdf")
    (generated_tmp_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    results = grade_directory(generated_tmp_dir)

    assert [result.source_pdf for result in results] == ["student-a.pdf", "student-b.pdf"]
    assert all(result.qr_data == DUMMY_QR_DATA for result in results)
    assert all(result.student_id == "63620147" for result in results)
    assert all(result.marked_answers["3"] == ["B", "C"] for result in results)
    assert all(result.omr_error == "" for result in results)


def test_grade_directory_continues_when_one_pdf_fails(generated_tmp_dir: Path, sample_pdfs: dict[str, Path]) -> None:
    shutil.copy(sample_pdfs["sample_answered"], generated_tmp_dir / "student-a.pdf")
    shutil.copy(sample_pdfs["missing_student_digit"], generated_tmp_dir / "student-b.pdf")

    results = grade_directory(generated_tmp_dir)

    assert [result.source_pdf for result in results] == ["student-a.pdf", "student-b.pdf"]
    assert results[0].student_id == "63620147"
    assert results[0].omr_error == ""
    assert results[1].student_id == ""
    assert results[1].marked_answers == {}
    assert "student-b.pdf" in results[1].omr_error
    assert "Student ID column 8 is empty" in results[1].omr_error


def test_grade_path_accepts_directory(generated_tmp_dir: Path, sample_pdfs: dict[str, Path]) -> None:
    shutil.copy(sample_pdfs["sample_answered"], generated_tmp_dir / "student-a.pdf")

    result = grade_path(generated_tmp_dir)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].source_pdf == "student-a.pdf"


def test_grade_rotated_answered_sample_pdf(sample_pdfs: dict[str, Path]) -> None:
    result = grade_pdf(sample_pdfs["rotated"])

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "33174026"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["B"],
        "2": ["B", "D"],
    }


def test_grade_translated_answered_sample_pdf(sample_pdfs: dict[str, Path]) -> None:
    result = grade_pdf(sample_pdfs["translated"])

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "33174026"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["B"],
        "2": ["B", "D"],
    }


def test_grade_pdf_writes_annotated_output(
    generated_tmp_dir: Path,
    sample_pdfs: dict[str, Path],
) -> None:
    output_pdf = generated_tmp_dir / "annotated.pdf"
    correct_answers = load_correct_answers(str(ANSWER_KEY_JSON))

    result = annotate_pdf(
        sample_pdfs["sample_answered"],
        output_path=output_pdf,
        correct_answers=correct_answers,
    )

    assert result.annotated_pdf == str(output_pdf)
    assert output_pdf.exists()
    text = PdfReader(str(output_pdf)).pages[0].extract_text()
    assert "Student ID: 63620147" in text
    assert DUMMY_QR_DATA["examSetId"] in text
    assert DUMMY_QR_DATA["variantId"] in text
    assert "QR Data:" not in text
    assert "Read Answers:" not in text


def test_load_correct_answers_from_json_file() -> None:
    assert load_correct_answers(str(ANSWER_KEY_JSON)) == {
        "1": ["D"],
        "3": ["B", "C"],
    }


def test_grade_directory_writes_annotated_outputs(
    generated_tmp_dir: Path,
    sample_pdfs: dict[str, Path],
) -> None:
    input_dir = generated_tmp_dir / "inputs"
    output_dir = generated_tmp_dir / "annotated"
    input_dir.mkdir()
    shutil.copy(sample_pdfs["sample_answered"], input_dir / "student-a.pdf")
    shutil.copy(sample_pdfs["markerless"], input_dir / "student-b.pdf")

    results = annotate_directory(input_dir, output_path=output_dir, correct_answers={"1": ["D"]})

    assert len(results) == 2
    assert Path(results[0].annotated_pdf).exists()
    assert Path(results[1].annotated_pdf).exists()
    assert results[0].annotated_pdf.endswith("student-a-annotated.pdf")
    assert results[1].annotated_pdf.endswith("student-b-annotated.pdf")
    error_text = PdfReader(results[1].annotated_pdf).pages[0].extract_text()
    assert "OMR Error:" in error_text


def test_correct_answer_overlay_skips_nonexistent_questions(
    generated_tmp_dir: Path,
    sample_pdfs: dict[str, Path],
) -> None:
    output_pdf = generated_tmp_dir / "translated-annotated.pdf"
    annotate_pdf(
        sample_pdfs["translated"],
        output_path=output_pdf,
        correct_answers=load_correct_answers(str(ANSWER_KEY_JSON)),
    )

    image = _rasterize_pdf_page(output_pdf)
    layout = PageLayout()

    def red_pixels_near(center_x_pt: float, center_y_pt: float) -> tuple[int, float, float]:
        scale_x = image.shape[1] / layout.page_width
        scale_y = image.shape[0] / layout.page_height
        center_x = int(round(center_x_pt * scale_x))
        center_y = int(round(image.shape[0] - (center_y_pt * scale_y)))
        patch = image[max(0, center_y - 10) : center_y + 11, max(0, center_x - 10) : center_x + 11]
        blue = patch[:, :, 0].astype(int)
        green = patch[:, :, 1].astype(int)
        red = patch[:, :, 2].astype(int)
        mask = (red > green + 15) & (red > blue + 15)
        red_y, red_x = mask.nonzero()
        if len(red_x) == 0:
            return 0, 0.0, 0.0
        return int(mask.sum()), float(red_x.mean() - 10), float(red_y.mean() - 10)

    def translated(center: tuple[float, float]) -> tuple[float, float]:
        return center[0] + 18.0, center[1] - 12.0

    question_1_d = translated(layout.answer_option_center(0, 0, OPTION_LABELS.index("D")))
    nominal_question_1_d = layout.answer_option_center(0, 0, OPTION_LABELS.index("D"))
    question_3_b = translated(layout.answer_option_center(0, 2, OPTION_LABELS.index("B")))

    red_count, red_dx, red_dy = red_pixels_near(*question_1_d)

    assert red_count > 0
    assert abs(red_dx) <= 1.0
    assert abs(red_dy) <= 1.0
    assert red_pixels_near(*nominal_question_1_d)[0] == 0
    assert red_pixels_near(*question_3_b)[0] == 0


def test_correct_answer_overlay_snaps_to_shifted_answer_bubbles(generated_tmp_dir: Path) -> None:
    layout = PageLayout()
    base_pdf = generated_tmp_dir / "base.pdf"
    shifted_pdf = generated_tmp_dir / "shifted.pdf"
    annotated_pdf = generated_tmp_dir / "shifted-annotated.pdf"
    shift_y = -4.0

    generate_omr_sheet(
        SheetConfig(
            question_count=10,
            choice_count=5,
            exam_set_id=TEST_EXAM_SET_ID,
            variant_id=TEST_VARIANT_ID,
        ),
        base_pdf,
    )
    _write_shifted_answer_rows_pdf(
        source_pdf=base_pdf,
        target_pdf=shifted_pdf,
        layout=layout,
        answers={1: ["B"]},
        student_id="00038145",
        row_count=10,
        shift_y=shift_y,
    )

    result = annotate_pdf(
        shifted_pdf,
        output_path=annotated_pdf,
        correct_answers={"1": ["B"]},
    )

    assert "B" in result.marked_answers["1"]

    image = _rasterize_pdf_page(annotated_pdf)
    center_x_pt, center_y_pt = layout.answer_option_center(0, 0, OPTION_LABELS.index("B"))
    scale_x = image.shape[1] / layout.page_width
    scale_y = image.shape[0] / layout.page_height
    center_x = int(round(center_x_pt * scale_x))
    center_y = int(round(image.shape[0] - ((center_y_pt + shift_y) * scale_y)))
    patch = image[max(0, center_y - 10) : center_y + 11, max(0, center_x - 10) : center_x + 11]
    blue = patch[:, :, 0].astype(int)
    green = patch[:, :, 1].astype(int)
    red = patch[:, :, 2].astype(int)
    mask = (red > green + 15) & (red > blue + 15)
    red_y, red_x = mask.nonzero()

    assert len(red_x) > 0
    assert abs(float(red_x.mean() - 10)) <= 1.0
    assert abs(float(red_y.mean() - 10)) <= 1.0


def test_bubble_center_refinement_ignores_hough_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    layout = PageLayout()
    image = np.full((int(layout.page_height), int(layout.page_width), 3), 255, dtype=np.uint8)
    alignment = _AlignedSheet(
        source_image=image,
        page_aligned_image=image,
        answer_aligned_image=image,
        qr_data=None,
        answer_aligned_to_source_transform=np.eye(3),
        source_image_width_px=image.shape[1],
        source_image_height_px=image.shape[0],
    )
    center_x = 100.0
    center_y = 100.0

    def hough_jitter(*args: object, **kwargs: object) -> np.ndarray:
        return np.array([[[25.0, 25.0, layout.bubble_radius]]], dtype=np.float32)

    monkeypatch.setattr(annotate_module.cv2, "HoughCircles", hough_jitter)

    assert _refine_source_bubble_center(
        layout=layout,
        alignment=alignment,
        center_x=center_x,
        center_y=center_y,
    ) == (center_x, center_y)

    def hough_real_shift(*args: object, **kwargs: object) -> np.ndarray:
        return np.array([[[30.0, 24.0, layout.bubble_radius]]], dtype=np.float32)

    monkeypatch.setattr(annotate_module.cv2, "HoughCircles", hough_real_shift)

    assert _refine_source_bubble_center(
        layout=layout,
        alignment=alignment,
        center_x=center_x,
        center_y=center_y,
    ) == pytest.approx((106.0, 100.0))


def test_grading_cli_outputs_json_and_annotation(
    generated_tmp_dir: Path,
    sample_pdfs: dict[str, Path],
    cli_env: dict[str, str],
) -> None:
    output_pdf = generated_tmp_dir / "cli-annotated.pdf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omr.annotate",
            str(sample_pdfs["sample_answered"]),
            "--output",
            str(output_pdf),
            "--correct-answers",
            str(ANSWER_KEY_JSON),
        ],
        cwd=ROOT,
        env=cli_env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload["student_id"] == "63620147"
    assert payload["annotated_pdf"] == str(output_pdf)
    assert output_pdf.exists()
