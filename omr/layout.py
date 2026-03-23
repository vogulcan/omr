from __future__ import annotations

from dataclasses import dataclass
from math import floor

from reportlab.lib.pagesizes import A4

from .models import SheetConfig

OPTION_LABELS = ("A", "B", "C", "D", "E")
STUDENT_ID_COLUMNS = 5
STUDENT_ID_ROWS = 10
MAX_QUESTIONS_PER_PAGE = 50


@dataclass(frozen=True, slots=True)
class PageLayout:
    page_width: float = A4[0]
    page_height: float = A4[1]
    margin: float = 36.0
    corner_marker_inset: float = 18.0
    corner_marker_size: float = 12.0
    header_title_gap: float = 22.0
    qr_size: float = 96.0
    qr_top_offset: float = 10.0
    qr_padding: float = 6.0
    student_id_top: float = 78.0
    student_id_label_width: float = 84.0
    student_id_column_gap: float = 18.0
    student_id_row_gap: float = 7.0
    student_id_digit_gap: float = 18.0
    questions_per_column: int = 13
    answer_top_gap: float = 34.0
    answer_column_gap: float = 15.0
    answer_row_height: float = 34.0
    answer_label_width: float = 18.0
    option_spacing: float = 16.0
    bubble_radius: float = 7.0
    local_marker_size: float = 12.0
    local_marker_y_gap: float = 30.0

    @property
    def bubble_diameter(self) -> float:
        return self.bubble_radius * 2

    @property
    def corner_marker_half_size(self) -> float:
        return self.corner_marker_size / 2

    @property
    def local_marker_half_size(self) -> float:
        return self.local_marker_size / 2

    @property
    def student_id_block_width(self) -> float:
        return (
            self.student_id_label_width
            + (STUDENT_ID_COLUMNS * self.bubble_diameter)
            + ((STUDENT_ID_COLUMNS - 1) * self.student_id_column_gap)
        )

    @property
    def student_id_block_height(self) -> float:
        return (STUDENT_ID_ROWS * self.bubble_diameter) + ((STUDENT_ID_ROWS - 1) * self.student_id_row_gap) + 24.0

    @property
    def question_block_width(self) -> float:
        return self.answer_label_width + (5 * self.option_spacing) + 16.0

    @property
    def answer_area_width(self) -> float:
        return self.page_width - (2 * self.margin)

    @property
    def answer_columns_per_page(self) -> int:
        usable_width = self.answer_area_width + self.answer_column_gap
        column_span = self.question_block_width + self.answer_column_gap
        return max(1, floor(usable_width / column_span))

    @property
    def questions_per_page(self) -> int:
        return min(self.answer_columns_per_page * self.questions_per_column, MAX_QUESTIONS_PER_PAGE)

    @property
    def qr_box_left(self) -> float:
        return self.page_width - self.margin - self.qr_size

    @property
    def qr_box_bottom(self) -> float:
        return self.page_height - self.margin - self.qr_size - self.qr_top_offset

    @property
    def qr_inner_left(self) -> float:
        return self.qr_box_left + self.qr_padding

    @property
    def qr_inner_bottom(self) -> float:
        return self.qr_box_bottom + self.qr_padding

    @property
    def qr_inner_size(self) -> float:
        return self.qr_size - (2 * self.qr_padding)

    @property
    def student_id_top_y(self) -> float:
        return self.page_height - self.student_id_top

    @property
    def student_id_bubble_top_y(self) -> float:
        return self.student_id_top_y - 20

    @property
    def student_id_bottom_y(self) -> float:
        return (
            self.page_height
            - self.student_id_top
            - 20
            - ((STUDENT_ID_ROWS - 1) * (self.bubble_diameter + self.student_id_row_gap))
            - self.bubble_radius
        )

    @property
    def answer_top_y(self) -> float:
        qr_bottom = self.qr_box_bottom
        return min(self.student_id_bottom_y, qr_bottom) - self.answer_top_gap

    @property
    def answer_left_x(self) -> float:
        return self.margin

    @property
    def answer_right_x(self) -> float:
        return self.margin + ((self.answer_columns_per_page - 1) * (self.question_block_width + self.answer_column_gap)) + self.question_block_width

    @property
    def answer_marker_y(self) -> float:
        return self.answer_top_y + self.local_marker_y_gap

    def corner_marker_centers(self) -> dict[str, tuple[float, float]]:
        center = self.corner_marker_inset + self.corner_marker_half_size
        return {
            "top_left": (center, self.page_height - center),
            "top_right": (self.page_width - center, self.page_height - center),
            "bottom_right": (self.page_width - center, center),
            "bottom_left": (center, center),
        }

    def local_marker_centers(self) -> dict[str, tuple[float, float]]:
        return {
            "answer_left": (self.answer_left_x + self.local_marker_half_size + 4.0, self.answer_marker_y),
            "answer_right": (self.answer_right_x - self.local_marker_half_size - 4.0, self.answer_marker_y),
        }

    def student_id_bubble_center(self, column_index: int, row_index: int) -> tuple[float, float]:
        center_x = (
            self.margin
            + self.student_id_label_width
            + column_index * (self.bubble_diameter + self.student_id_column_gap)
        )
        center_y = self.student_id_bubble_top_y - row_index * (self.bubble_diameter + self.student_id_row_gap)
        return center_x, center_y

    def answer_option_center(self, column_index: int, row_index: int, option_index: int) -> tuple[float, float]:
        center_x = (
            self.margin
            + column_index * (self.question_block_width + self.answer_column_gap)
            + self.answer_label_width
            + 12
            + option_index * self.option_spacing
        )
        center_y = self.answer_top_y - row_index * self.answer_row_height + 2
        return center_x, center_y


@dataclass(frozen=True, slots=True)
class QuestionPlacement:
    question_number: int
    option_count: int
    page_index: int
    column_index: int
    row_index: int


def paginate_questions(config: SheetConfig, layout: PageLayout | None = None) -> list[list[QuestionPlacement]]:
    layout = layout or PageLayout()
    pages: list[list[QuestionPlacement]] = []

    for question_index, option_count in enumerate(config.question_option_counts):
        page_index = question_index // layout.questions_per_page
        position_in_page = question_index % layout.questions_per_page
        column_index = position_in_page // layout.questions_per_column
        row_index = position_in_page % layout.questions_per_column

        while len(pages) <= page_index:
            pages.append([])

        pages[page_index].append(
            QuestionPlacement(
                question_number=question_index + 1,
                option_count=option_count,
                page_index=page_index,
                column_index=column_index,
                row_index=row_index,
            )
        )

    return pages
