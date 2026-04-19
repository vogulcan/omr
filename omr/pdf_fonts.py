from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import EmbeddedType1Face, Font
from reportlab.pdfbase.ttfonts import TTFont


@dataclass(frozen=True, slots=True)
class PdfFontSet:
    regular: str
    bold: str


@dataclass(frozen=True, slots=True)
class _FontCandidate:
    alias: str
    ttf_names: tuple[str, ...]
    type1_names: tuple[str, str]
    package_key: str


_STYLE_SPECS = {
    "regular": (
        _FontCandidate(
            alias="OMRLatinModern-Regular",
            ttf_names=("lmroman10-regular.ttf",),
            type1_names=("lmr10.afm", "lmr10.pfb"),
            package_key="lm",
        ),
        _FontCandidate(
            alias="OMRComputerModern-Regular",
            ttf_names=("cmunrm.ttf",),
            type1_names=("cmr10.afm", "cmr10.pfb"),
            package_key="amsfonts",
        ),
    ),
    "bold": (
        _FontCandidate(
            alias="OMRLatinModern-Bold",
            ttf_names=("lmroman10-bold.ttf",),
            type1_names=("lmbx10.afm", "lmbx10.pfb"),
            package_key="lm",
        ),
        _FontCandidate(
            alias="OMRComputerModern-Bold",
            ttf_names=("cmunbx.ttf",),
            type1_names=("cmbx10.afm", "cmbx10.pfb"),
            package_key="amsfonts",
        ),
    ),
}

_PACKAGE_DOWNLOADS = {
    "lm": {
        "url": "https://mirrors.ctan.org/fonts/lm.zip",
        "members": {
            "lmr10.afm": "lm/fonts/afm/public/lm/lmr10.afm",
            "lmr10.pfb": "lm/fonts/type1/public/lm/lmr10.pfb",
            "lmbx10.afm": "lm/fonts/afm/public/lm/lmbx10.afm",
            "lmbx10.pfb": "lm/fonts/type1/public/lm/lmbx10.pfb",
        },
    },
    "amsfonts": {
        "url": "https://mirrors.ctan.org/fonts/amsfonts.zip",
        "members": {
            "cmr10.afm": "amsfonts/afm/cmr10.afm",
            "cmr10.pfb": "amsfonts/pfb/cmr10.pfb",
            "cmbx10.afm": "amsfonts/afm/cmbx10.afm",
            "cmbx10.pfb": "amsfonts/pfb/cmbx10.pfb",
        },
    },
}


@lru_cache(maxsize=1)
def get_pdf_fonts() -> PdfFontSet:
    return PdfFontSet(
        regular=_register_style("regular"),
        bold=_register_style("bold"),
    )


def _register_style(style: str) -> str:
    for candidate in _STYLE_SPECS[style]:
        system_font = _resolve_system_font(candidate)
        if system_font is not None:
            _register_font(candidate.alias, system_font)
            return candidate.alias

        cached_font = _resolve_cached_font(candidate)
        if cached_font is not None:
            _register_font(candidate.alias, cached_font)
            return candidate.alias

    raise RuntimeError(
        "Unable to resolve Latin Modern or Computer Modern fonts from the system "
        "or from the local cache download."
    )


def _resolve_system_font(candidate: _FontCandidate) -> tuple[str, Path] | tuple[str, Path, Path] | None:
    index = _system_font_index()

    for ttf_name in candidate.ttf_names:
        ttf_path = index.get(ttf_name)
        if ttf_path is not None:
            return ("ttf", ttf_path)

    afm_name, pfb_name = candidate.type1_names
    afm_path = index.get(afm_name)
    pfb_path = index.get(pfb_name)
    if afm_path is not None and pfb_path is not None:
        return ("type1", afm_path, pfb_path)

    return None


def _resolve_cached_font(candidate: _FontCandidate) -> tuple[str, Path, Path] | None:
    cache_dir = _font_cache_dir()
    afm_name, pfb_name = candidate.type1_names
    afm_path = cache_dir / afm_name
    pfb_path = cache_dir / pfb_name
    if afm_path.exists() and pfb_path.exists():
        return ("type1", afm_path, pfb_path)

    _download_package_fonts(candidate.package_key, cache_dir)
    if afm_path.exists() and pfb_path.exists():
        return ("type1", afm_path, pfb_path)
    return None


def _register_font(alias: str, font_spec: tuple[str, Path] | tuple[str, Path, Path]) -> None:
    if alias in pdfmetrics.getRegisteredFontNames():
        return

    kind = font_spec[0]
    if kind == "ttf":
        _, ttf_path = font_spec
        pdfmetrics.registerFont(TTFont(alias, str(ttf_path)))
        return

    _, afm_path, pfb_path = font_spec
    face = EmbeddedType1Face(str(afm_path), str(pfb_path))
    pdfmetrics.registerTypeFace(face)
    pdfmetrics.registerFont(Font(alias, face.name, "WinAnsiEncoding"))


def _download_package_fonts(package_key: str, cache_dir: Path) -> None:
    package = _PACKAGE_DOWNLOADS[package_key]
    expected_files = [cache_dir / name for name in package["members"]]
    if all(path.exists() for path in expected_files):
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omr-fonts-") as temp_dir:
        archive_path = Path(temp_dir) / f"{package_key}.zip"
        with urllib.request.urlopen(package["url"], timeout=60) as response, archive_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

        with zipfile.ZipFile(archive_path) as archive:
            for target_name, member_name in package["members"].items():
                target_path = cache_dir / target_name
                if target_path.exists():
                    continue
                with archive.open(member_name) as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)


@lru_cache(maxsize=1)
def _system_font_index() -> dict[str, Path]:
    target_names = {
        name
        for candidates in _STYLE_SPECS.values()
        for candidate in candidates
        for name in (*candidate.ttf_names, *candidate.type1_names)
    }
    roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local" / "share" / "fonts",
        Path.home() / ".fonts",
        Path("/usr/share/texmf-dist"),
        Path("/usr/share/texlive"),
    ]

    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename in target_names and filename not in index:
                    index[filename] = Path(dirpath) / filename
    return index


def _font_cache_dir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "omr" / "fonts"
