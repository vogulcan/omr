from __future__ import annotations

from dataclasses import dataclass

MAX_QUESTION_COUNT = 100


@dataclass(slots=True)
class SheetConfig:
    question_count: int = 0
    choice_count: int = 0
    exam_set_id: str = ""
    variant_id: str = ""
    title: str = "Optical Mark Recognition Sheet"
    instructions: str = "Fill bubbles completely."

    @property
    def question_option_counts(self) -> list[int]:
        return [self.choice_count] * self.question_count

    def __post_init__(self) -> None:
        if self.question_count < 1:
            raise ValueError("question_count must be at least 1")
        if self.question_count > MAX_QUESTION_COUNT:
            raise ValueError(f"question_count must not exceed {MAX_QUESTION_COUNT}")

        if self.choice_count < 2 or self.choice_count > 5:
            raise ValueError("choice_count must be between 2 and 5")

        if not self.exam_set_id.strip():
            raise ValueError("exam_set_id must not be empty")

        if not self.variant_id.strip():
            raise ValueError("variant_id must not be empty")
