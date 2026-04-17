from __future__ import annotations

import io
import os
import shutil
import sys
from pathlib import Path

import cv2
import pytest
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.colors import black, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omr.generator import DUMMY_QR_DATA, generate_omr_sheet
from omr.grade import _rasterize_pdf_page
from omr.layout import OPTION_LABELS, PageLayout
from omr.models import SheetConfig


def _sanitize_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in value)


def _merge_overlay(source_pdf: Path, overlay_pdf_bytes: bytes, target_pdf: Path) -> None:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    overlay_reader = PdfReader(io.BytesIO(overlay_pdf_bytes))
    for page in reader.pages:
        writer.add_page(page)
    writer.pages[0].merge_page(overlay_reader.pages[0])
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _mark_sheet(
    *,
    source_pdf: Path,
    target_pdf: Path,
    layout: PageLayout,
    student_id: str,
    answers: dict[str, list[str]],
    weak_student_digit_columns: set[int] | None = None,
    weak_answers: set[tuple[str, str]] | None = None,
) -> None:
    weak_student_digit_columns = weak_student_digit_columns or set()
    weak_answers = weak_answers or set()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(layout.page_width, layout.page_height))
    pdf.setFillColor(black)
    pdf.setStrokeColor(black)

    for column_index, digit in enumerate(student_id):
        center_x, center_y = layout.student_id_bubble_center(column_index, int(digit))
        radius = layout.bubble_radius * (0.38 if column_index in weak_student_digit_columns else 0.58)
        pdf.circle(center_x, center_y, radius, stroke=0, fill=1)

    for question_key, labels in answers.items():
        question_number = int(question_key)
        placement_index = question_number - 1
        column_index = (placement_index % layout.questions_per_page) // layout.questions_per_column
        row_index = placement_index % layout.questions_per_column
        for label in labels:
            option_index = OPTION_LABELS.index(label)
            center_x, center_y = layout.answer_option_center(column_index, row_index, option_index)
            radius = layout.bubble_radius * (0.42 if (question_key, label) in weak_answers else 0.58)
            pdf.circle(center_x, center_y, radius, stroke=0, fill=1)

    pdf.save()
    _merge_overlay(source_pdf, buffer.getvalue(), target_pdf)


def _erase_top_right_marker(source_pdf: Path, target_pdf: Path, layout: PageLayout) -> None:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(layout.page_width, layout.page_height))
    center_x, center_y = layout.corner_marker_centers()["top_right"]
    erase_size = layout.corner_marker_size + 10
    pdf.setFillColor(white)
    pdf.setStrokeColor(white)
    pdf.rect(
        center_x - erase_size / 2,
        center_y - erase_size / 2,
        erase_size,
        erase_size,
        stroke=0,
        fill=1,
    )
    pdf.save()
    _merge_overlay(source_pdf, buffer.getvalue(), target_pdf)


def _write_image_pdf(image_bgr, target_pdf: Path, layout: PageLayout) -> None:
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("Failed to encode transformed test image")

    pdf = canvas.Canvas(str(target_pdf), pagesize=(layout.page_width, layout.page_height))
    pdf.drawImage(
        ImageReader(io.BytesIO(encoded.tobytes())),
        0,
        0,
        width=layout.page_width,
        height=layout.page_height,
    )
    pdf.save()


def _transform_pdf(source_pdf: Path, target_pdf: Path, layout: PageLayout, *, angle_degrees: float, shift_x: float, shift_y: float) -> None:
    image = _rasterize_pdf_page(source_pdf)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle_degrees, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    _write_image_pdf(transformed, target_pdf, layout)


def _translate_pdf_vector(source_pdf: Path, target_pdf: Path, *, shift_x: float, shift_y: float) -> None:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page in reader.pages:
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        target_page = writer.add_blank_page(width=width, height=height)
        target_page.merge_transformed_page(page, Transformation().translate(tx=shift_x, ty=shift_y))
    with target_pdf.open("wb") as handle:
        writer.write(handle)


def _make_base_sheet(target_pdf: Path, question_count: int, choice_count: int, *, exam_set_id: str, variant_id: str) -> None:
    generate_omr_sheet(
        SheetConfig(
            question_count=question_count,
            choice_count=choice_count,
            exam_set_id=exam_set_id,
            variant_id=variant_id,
        ),
        target_pdf,
    )


@pytest.fixture(scope="session")
def generated_pdf_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    generated_root = tmp_path_factory.mktemp("omr-generated")
    yield generated_root
    shutil.rmtree(generated_root, ignore_errors=True)


@pytest.fixture()
def generated_tmp_dir(generated_pdf_dir: Path, request: pytest.FixtureRequest) -> Path:
    target = generated_pdf_dir / _sanitize_name(request.node.name)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


@pytest.fixture(scope="session")
def sample_pdfs(generated_pdf_dir: Path) -> dict[str, Path]:
    layout = PageLayout()

    sample_base = generated_pdf_dir / "sample-base.pdf"
    sample_answered = generated_pdf_dir / "sample-answered.pdf"
    _make_base_sheet(
        sample_base,
        6,
        5,
        exam_set_id=DUMMY_QR_DATA["examSetId"],
        variant_id=DUMMY_QR_DATA["variantId"],
    )
    _mark_sheet(
        source_pdf=sample_base,
        target_pdf=sample_answered,
        layout=layout,
        student_id="63620",
        answers={
            "1": ["D"],
            "2": ["C"],
            "3": ["B", "C"],
            "4": ["C"],
            "5": ["A"],
            "6": ["E"],
        },
    )

    answer1_base = generated_pdf_dir / "answer1-base.pdf"
    answer1_pdf = generated_pdf_dir / "answer1.pdf"
    missing_student_digit_pdf = generated_pdf_dir / "missing-student-digit.pdf"
    _make_base_sheet(
        answer1_base,
        6,
        5,
        exam_set_id=DUMMY_QR_DATA["examSetId"],
        variant_id=DUMMY_QR_DATA["variantId"],
    )
    _mark_sheet(
        source_pdf=answer1_base,
        target_pdf=answer1_pdf,
        layout=layout,
        student_id="01345",
        answers={
            "1": ["D"],
            "2": ["C"],
            "3": ["B", "D"],
            "4": ["C"],
            "5": ["B"],
            "6": ["D"],
        },
        weak_student_digit_columns={4},
        weak_answers={("3", "D"), ("6", "D")},
    )
    _mark_sheet(
        source_pdf=answer1_base,
        target_pdf=missing_student_digit_pdf,
        layout=layout,
        student_id="0134",
        answers={
            "1": ["D"],
            "2": ["C"],
            "3": ["B", "D"],
            "4": ["C"],
            "5": ["B"],
            "6": ["D"],
        },
        weak_answers={("3", "D"), ("6", "D")},
    )

    markerless_pdf = generated_pdf_dir / "markerless.pdf"
    _erase_top_right_marker(sample_answered, markerless_pdf, layout)

    rotated_base = generated_pdf_dir / "rotated-base.pdf"
    _make_base_sheet(
        rotated_base,
        2,
        4,
        exam_set_id=DUMMY_QR_DATA["examSetId"],
        variant_id=DUMMY_QR_DATA["variantId"],
    )
    rotated_answered_base = generated_pdf_dir / "rotated-answered-base.pdf"
    _mark_sheet(
        source_pdf=rotated_base,
        target_pdf=rotated_answered_base,
        layout=layout,
        student_id="33174",
        answers={
            "1": ["B"],
            "2": ["B", "D"],
        },
    )

    rotated_pdf = generated_pdf_dir / "rotated.pdf"
    translated_pdf = generated_pdf_dir / "translated.pdf"
    _transform_pdf(rotated_answered_base, rotated_pdf, layout, angle_degrees=2.4, shift_x=6, shift_y=-10)
    _translate_pdf_vector(rotated_answered_base, translated_pdf, shift_x=18, shift_y=-12)

    return {
        "sample_answered": sample_answered,
        "answer1": answer1_pdf,
        "missing_student_digit": missing_student_digit_pdf,
        "markerless": markerless_pdf,
        "rotated": rotated_pdf,
        "translated": translated_pdf,
    }


@pytest.fixture()
def cli_env() -> dict[str, str]:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) if not existing_pythonpath else f"{ROOT}:{existing_pythonpath}"
    return env
