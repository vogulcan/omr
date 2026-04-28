from __future__ import annotations

import json
from pathlib import Path
from typing import BinaryIO

from reportlab.lib.colors import black
from reportlab.pdfgen import canvas
import segno

from .layout import OPTION_LABELS, STUDENT_ID_COLUMNS, STUDENT_ID_ROWS, PageLayout, paginate_questions
from .models import SheetConfig
from .pdf_fonts import PdfFontSet, get_pdf_fonts

DUMMY_QR_DATA = {
    "examSetId": "f6adcc63-71dc-412c-9c8d-a4609df454ff",
    "variantId": "37e3d65f-e540-4e34-b438-549e731be3b0",
}


def generate_omr_sheet(
    config: SheetConfig,
    destination: str | Path | BinaryIO,
    layout: PageLayout | None = None,
) -> None:
    layout = layout or PageLayout()
    fonts = get_pdf_fonts()

    pdf = canvas.Canvas(str(destination) if isinstance(destination, (str, Path)) else destination)
    pdf.setPageCompression(0)

    for page_index, page_questions in enumerate(paginate_questions(config, layout), start=1):
        _draw_page(pdf, config, layout, fonts, page_questions, page_index)
        pdf.showPage()

    pdf.save()


def _draw_page(
    pdf: canvas.Canvas,
    config: SheetConfig,
    layout: PageLayout,
    fonts: PdfFontSet,
    page_questions,
    page_number: int,
) -> None:
    page_width = layout.page_width
    page_height = layout.page_height
    margin = layout.margin

    pdf.setPageSize((page_width, page_height))
    pdf.setStrokeColor(black)
    pdf.setFillColor(black)

    _draw_alignment_markers(pdf, layout)
    _draw_header(pdf, config, layout, fonts, page_number)
    _draw_qr_placeholder(pdf, layout, config)
    _draw_student_id_block(pdf, layout, fonts)
    _draw_handwritten_info_block(pdf, layout, fonts)
    _draw_question_area(pdf, layout, fonts, page_questions)


def _draw_header(pdf: canvas.Canvas, config: SheetConfig, layout: PageLayout, fonts: PdfFontSet, page_number: int) -> None:
    top = layout.page_height - layout.margin
    left = layout.margin

    pdf.setFont(fonts.bold, 18)
    pdf.drawString(left, top, config.title)

    pdf.setFont(fonts.regular, 10)
    pdf.drawString(left, top - layout.header_title_gap, config.instructions)


def _draw_qr_placeholder(pdf: canvas.Canvas, layout: PageLayout, config: SheetConfig) -> None:
    x = layout.qr_box_left
    y = layout.qr_box_bottom
    padding = layout.qr_padding
    qr = segno.make(dummy_qr_payload(config), error="m")
    matrix = tuple(tuple(int(cell) for cell in row) for row in qr.matrix)
    module_rows = len(matrix)
    module_cols = len(matrix[0]) if matrix else 0
    module_size = min(
        (layout.qr_size - (2 * padding)) / module_cols,
        (layout.qr_size - (2 * padding)) / module_rows,
    )
    qr_width = module_cols * module_size
    qr_height = module_rows * module_size
    qr_x = x + (layout.qr_size - qr_width) / 2
    qr_y = y + (layout.qr_size - qr_height) / 2

    pdf.rect(x, y, layout.qr_size, layout.qr_size)
    for row_index, row in enumerate(matrix):
        for column_index, cell in enumerate(row):
            if not cell:
                continue
            module_x = qr_x + column_index * module_size
            module_y = qr_y + (module_rows - row_index - 1) * module_size
            pdf.rect(module_x, module_y, module_size, module_size, stroke=0, fill=1)


def _draw_student_id_block(pdf: canvas.Canvas, layout: PageLayout, fonts: PdfFontSet) -> None:
    top = layout.student_id_top_y
    left = layout.margin

    pdf.setFont(fonts.bold, 12)
    pdf.drawString(left, top, "Student ID")

    bubble_top = layout.student_id_bubble_top_y
    digit_label_x = left + 2
    first_column_x = left + layout.student_id_label_width

    pdf.setFont(fonts.regular, 10)
    for row_index in range(STUDENT_ID_ROWS):
        row_y = bubble_top - row_index * (layout.bubble_diameter + layout.student_id_row_gap)
        pdf.drawRightString(digit_label_x + 18, row_y - 3, str(row_index))

        for column_index in range(STUDENT_ID_COLUMNS):
            column_x = first_column_x + column_index * (layout.bubble_diameter + layout.student_id_column_gap)
            pdf.circle(column_x, row_y, layout.bubble_radius)

    header_y = bubble_top + 16
    pdf.setFont(fonts.regular, 10)
    for column_index in range(STUDENT_ID_COLUMNS):
        column_x = first_column_x + column_index * (layout.bubble_diameter + layout.student_id_column_gap)
        pdf.drawCentredString(column_x, header_y, str(column_index + 1))


def _draw_handwritten_info_block(pdf: canvas.Canvas, layout: PageLayout, fonts: PdfFontSet) -> None:
    left = layout.handwritten_block_left
    right = layout.handwritten_block_right
    top = layout.handwritten_block_top_y
    bottom = layout.handwritten_block_bottom_y

    if right <= left or top <= bottom:
        return

    width = right - left
    height = top - bottom
    row_height = height / 3.0
    fields = ("Name", "ID", "Signature")

    pdf.saveState()
    pdf.roundRect(left, bottom, width, height, 6, stroke=1, fill=0)
    pdf.setFont(fonts.bold, 9)
    for row_index, label in enumerate(fields):
        row_top = top - row_index * row_height
        row_bottom = row_top - row_height
        if row_index:
            pdf.line(left, row_top, right, row_top)
        pdf.drawString(left + 10, row_top - 14, label)
        pdf.line(left + 10, row_bottom + 12, right - 10, row_bottom + 12)
    pdf.restoreState()


def _draw_question_area(pdf: canvas.Canvas, layout: PageLayout, fonts: PdfFontSet, page_questions) -> None:
    answer_top_y = layout.answer_top_y
    left = layout.margin

    pdf.setFont(fonts.regular, 8)
    for placement in page_questions:
        column_x = left + placement.column_index * (layout.question_block_width + layout.answer_column_gap)
        row_y = answer_top_y - placement.row_index * layout.answer_row_height

        pdf.setFont(fonts.bold, 8)
        pdf.drawRightString(column_x + layout.answer_label_width, row_y - 1, f"{placement.question_number}.")

        pdf.setFont(fonts.regular, 7)
        for option_index in range(placement.option_count):
            bubble_x, bubble_y = layout.answer_option_center(
                placement.column_index,
                placement.row_index,
                option_index,
            )
            pdf.circle(bubble_x, bubble_y, layout.bubble_radius)
            pdf.drawCentredString(bubble_x, bubble_y + 6, OPTION_LABELS[option_index])


def _draw_alignment_markers(pdf: canvas.Canvas, layout: PageLayout) -> None:
    for center_x, center_y in layout.corner_marker_centers().values():
        pdf.rect(
            center_x - layout.corner_marker_half_size,
            center_y - layout.corner_marker_half_size,
            layout.corner_marker_size,
            layout.corner_marker_size,
            stroke=0,
            fill=1,
        )

    for center_x, center_y in layout.local_marker_centers().values():
        pdf.rect(
            center_x - layout.local_marker_half_size,
            center_y - layout.local_marker_half_size,
            layout.local_marker_size,
            layout.local_marker_size,
            stroke=0,
            fill=1,
        )


def dummy_qr_payload(config: SheetConfig | None = None) -> str:
    payload = DUMMY_QR_DATA
    if config is not None:
        payload = {
            "examSetId": config.exam_set_id,
            "variantId": config.variant_id,
        }
    return json.dumps(payload, separators=(",", ":"))
