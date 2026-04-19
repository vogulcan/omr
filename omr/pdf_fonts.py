from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import EmbeddedType1Face, Font

FONT_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


@dataclass(frozen=True, slots=True)
class PdfFontSet:
    regular: str
    bold: str


_STYLE_SPECS = {
    "regular": (
        ("OMRLatinModern-Regular", "lmr10.afm", "lmr10.pfb"),
        ("OMRComputerModern-Regular", "cmr10.afm", "cmr10.pfb"),
    ),
    "bold": (
        ("OMRLatinModern-Bold", "lmbx10.afm", "lmbx10.pfb"),
        ("OMRComputerModern-Bold", "cmbx10.afm", "cmbx10.pfb"),
    ),
}


@lru_cache(maxsize=1)
def get_pdf_fonts() -> PdfFontSet:
    return PdfFontSet(
        regular=_register_style("regular"),
        bold=_register_style("bold"),
    )


def _register_style(style: str) -> str:
    for alias, afm_name, pfb_name in _STYLE_SPECS[style]:
        afm_path = FONT_ASSET_DIR / afm_name
        pfb_path = FONT_ASSET_DIR / pfb_name
        if not afm_path.exists() or not pfb_path.exists():
            continue
        _register_type1_font(alias, afm_path, pfb_path)
        return alias
    raise RuntimeError(f"No usable PDF font asset found for style '{style}'")


def _register_type1_font(alias: str, afm_path: Path, pfb_path: Path) -> None:
    if alias in pdfmetrics.getRegisteredFontNames():
        return

    face = EmbeddedType1Face(str(afm_path), str(pfb_path))
    pdfmetrics.registerTypeFace(face)
    pdfmetrics.registerFont(Font(alias, face.name, "WinAnsiEncoding"))
