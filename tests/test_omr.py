from __future__ import annotations

import re
from pathlib import Path
import shutil

import pytest

from omr.cli import parse_question_counts
from omr.grade import (
    UnsupportedSheetError,
    _detect_answer_marker_centers,
    _detect_page_marker_centers,
    _rasterize_pdf_page,
    _threshold_image,
    grade_directory,
    grade_path,
    grade_pdf,
)
from omr.generator import DUMMY_QR_DATA, dummy_qr_payload, generate_omr_sheet
from omr.layout import OPTION_LABELS, STUDENT_ID_COLUMNS, STUDENT_ID_ROWS, PageLayout, paginate_questions
from omr.models import SheetConfig

SAMPLE_ANSWERED_PDF = Path(__file__).resolve().parents[1] / "answered-sheets" / "omr-sheet-answered.pdf"
ANSWER1_PDF = Path(__file__).resolve().parents[1] / "answered-sheets" / "answer1.pdf"
UNSUPPORTED_MARKERLESS_PDF = Path(__file__).resolve().parents[1] / "answered-sheets" / "omr-sheet-answered2.pdf"
ROTATED_ANSWERED_PDF = Path(__file__).resolve().parents[1] / "answered-sheets" / "omr-sheet-answered5.pdf"
TRANSLATED_ANSWERED_PDF = Path(__file__).resolve().parents[1] / "answered-sheets" / "omr-sheet-answered6.pdf"


def test_config_rejects_empty_question_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SheetConfig(question_option_counts=[])


@pytest.mark.parametrize("invalid_count", [0, 1, 6])
def test_config_rejects_invalid_question_option_counts(invalid_count: int) -> None:
    with pytest.raises(ValueError, match="between 2 and 5"):
        SheetConfig(question_option_counts=[4, invalid_count, 5])


def test_config_accepts_mixed_valid_counts() -> None:
    config = SheetConfig(question_option_counts=[2, 3, 4, 5])
    assert config.question_option_counts == [2, 3, 4, 5]


def test_student_id_area_dimensions_are_fixed() -> None:
    layout = PageLayout()
    assert STUDENT_ID_COLUMNS == 5
    assert STUDENT_ID_ROWS == 10
    assert layout.student_id_block_width > 0
    assert layout.student_id_block_height > 0


def test_question_pagination_caps_at_ten_rows_per_column() -> None:
    config = SheetConfig(question_option_counts=[4] * 12)
    pages = paginate_questions(config)

    assert len(pages) == 1
    assert pages[0][9].column_index == 0
    assert pages[0][9].row_index == 9
    assert pages[0][10].column_index == 1
    assert pages[0][10].row_index == 0


def test_question_option_labels_match_choice_count() -> None:
    counts = [2, 3, 4, 5]
    config = SheetConfig(question_option_counts=counts)
    page = paginate_questions(config)[0]

    for placement, expected_count in zip(page, counts, strict=True):
        assert placement.option_count == expected_count
        assert list(OPTION_LABELS[:expected_count]) == list(OPTION_LABELS[: placement.option_count])


def test_pagination_occurs_when_page_capacity_is_exceeded() -> None:
    layout = PageLayout()
    config = SheetConfig(question_option_counts=[5] * (layout.questions_per_page + 1))
    pages = paginate_questions(config, layout)

    assert len(pages) == 2
    assert len(pages[0]) == layout.questions_per_page
    assert pages[1][0].question_number == layout.questions_per_page + 1


def test_parse_question_counts() -> None:
    assert parse_question_counts("2, 3,4,5") == [2, 3, 4, 5]


def test_dummy_qr_payload_is_expected_json() -> None:
    assert dummy_qr_payload() == (
        '{"examSetId":"f6adcc63-71dc-412c-9c8d-a4609df454ff",'
        '"variantId":"37e3d65f-e540-4e34-b438-549e731be3b0"}'
    )
    assert DUMMY_QR_DATA["examSetId"]
    assert DUMMY_QR_DATA["variantId"]


def test_generate_single_page_pdf(tmp_path: Path) -> None:
    target = tmp_path / "single-page.pdf"
    generate_omr_sheet(SheetConfig(question_option_counts=[4] * 20), target)

    data = target.read_bytes()
    assert target.exists()
    assert b"Page 1" not in data
    assert b"Dummy QR code" not in data
    assert len(re.findall(rb"/Type /Page\b", data)) == 1


def test_generated_sheet_contains_detectable_markers(tmp_path: Path) -> None:
    target = tmp_path / "marker-sheet.pdf"
    layout = PageLayout()
    generate_omr_sheet(SheetConfig(question_option_counts=[4] * 6), target)

    image = _rasterize_pdf_page(target)
    binary = _threshold_image(image)

    corner_markers = _detect_page_marker_centers(binary, layout)
    local_markers = _detect_answer_marker_centers(binary, layout)

    assert corner_markers.shape == (4, 2)
    assert local_markers.shape == (2, 2)


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


def test_generate_multi_page_pdf(tmp_path: Path) -> None:
    layout = PageLayout()
    target = tmp_path / "multi-page.pdf"
    generate_omr_sheet(SheetConfig(question_option_counts=[5] * (layout.questions_per_page + 7)), target)

    data = target.read_bytes()
    assert len(re.findall(rb"/Type /Page\b", data)) == 2
    assert b"Dummy QR code" not in data


def test_grade_markerless_sheet_fails_cleanly() -> None:
    with pytest.raises(UnsupportedSheetError, match="alignment marker"):
        grade_pdf(UNSUPPORTED_MARKERLESS_PDF)


def test_grade_answered_sample_pdf() -> None:
    result = grade_pdf(SAMPLE_ANSWERED_PDF)

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "63620"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["D"],
        "2": ["C"],
        "3": ["B", "C"],
        "4": ["C"],
        "5": ["A"],
        "6": ["E"],
    }


def test_grade_answer1_pdf() -> None:
    result = grade_pdf(ANSWER1_PDF)

    assert result.student_id == "01345"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["D"],
        "2": ["C"],
        "3": ["B", "D"],
        "4": ["C"],
        "5": ["B"],
        "6": ["D"],
    }


def test_grade_directory_reads_all_pdfs(tmp_path: Path) -> None:
    shutil.copy(SAMPLE_ANSWERED_PDF, tmp_path / "student-a.pdf")
    shutil.copy(SAMPLE_ANSWERED_PDF, tmp_path / "student-b.pdf")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    results = grade_directory(tmp_path)

    assert [result.source_pdf for result in results] == ["student-a.pdf", "student-b.pdf"]
    assert all(result.qr_data == DUMMY_QR_DATA for result in results)
    assert all(result.student_id == "63620" for result in results)
    assert all(result.marked_answers["3"] == ["B", "C"] for result in results)
    assert all(result.omr_error == "" for result in results)


def test_grade_directory_continues_when_one_pdf_fails(tmp_path: Path) -> None:
    shutil.copy(SAMPLE_ANSWERED_PDF, tmp_path / "student-a.pdf")
    shutil.copy(UNSUPPORTED_MARKERLESS_PDF, tmp_path / "student-b.pdf")

    results = grade_directory(tmp_path)

    assert [result.source_pdf for result in results] == ["student-a.pdf", "student-b.pdf"]
    assert results[0].student_id == "63620"
    assert results[0].omr_error == ""
    assert results[1].student_id == ""
    assert results[1].marked_answers == {}
    assert "student-b.pdf" in results[1].omr_error
    assert "alignment marker" in results[1].omr_error


def test_grade_path_accepts_directory(tmp_path: Path) -> None:
    shutil.copy(SAMPLE_ANSWERED_PDF, tmp_path / "student-a.pdf")

    result = grade_path(tmp_path)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].source_pdf == "student-a.pdf"


def test_grade_rotated_answered_sample_pdf() -> None:
    result = grade_pdf(ROTATED_ANSWERED_PDF)

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "33174"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["B"],
        "2": ["B", "D"],
    }


def test_grade_translated_answered_sample_pdf() -> None:
    result = grade_pdf(TRANSLATED_ANSWERED_PDF)

    assert result.qr_data == DUMMY_QR_DATA
    assert result.student_id == "33174"
    assert result.omr_error == ""
    assert result.marked_answers == {
        "1": ["B"],
        "2": ["B", "D"],
    }
