from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SheetConfig:
    question_option_counts: list[int] = field(default_factory=list)
    title: str = "Optical Mark Recognition Sheet"
    instructions: str = "Fill bubbles completely."

    def __post_init__(self) -> None:
        if not self.question_option_counts:
            raise ValueError("question_option_counts must not be empty")

        invalid_counts = [count for count in self.question_option_counts if count < 2 or count > 5]
        if invalid_counts:
            raise ValueError("Each question option count must be between 2 and 5")
