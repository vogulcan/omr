# OMR Sheet Generator

`omr` generates marker-based OMR sheets as PDF files.
`omr-grade` reads filled sheets and returns JSON.
`omr-annotate` writes annotated PDFs with grading metadata and optional answer-key overlays.

## What It Does

Generated sheets have:

- A4 portrait layout
- 4 corner registration markers
- 2 local answer-area registration markers
- a 5-digit student ID block
- question columns with at most `10` questions per column
- `2` to `5` options per question
- a QR code containing `examSetId` and `variantId`

Grading returns:

- QR data
- student ID
- marked answers
- per-file error text when grading fails

Annotation writes:

- a copy of the original PDF with red watermark text below the QR
- optional faint red correct-answer overlays on detected question rows

## Requirements

- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)
- Poppler installed so `pdftoppm` is available on `PATH`

## Install

```bash
uv sync
```

Useful help commands:

```bash
uv run omr --help
uv run omr-grade --help
uv run omr-annotate --help
```

## Typical Workflow

1. Generate a blank sheet.
2. Print and fill it.
3. Grade the filled PDF to JSON.
4. Optionally annotate the filled PDF for review.

## Generate a Sheet

Minimal example:

```bash
uv run omr \
  --questions 4,4,5,3,2,5 \
  --exam-set-id f6adcc63-71dc-412c-9c8d-a4609df454ff \
  --variant-id 37e3d65f-e540-4e34-b438-549e731be3b0 \
  --output omr-sheet.pdf
```

With custom title and instructions:

```bash
uv run omr \
  --questions 4,4,4,4,5,5,3,2 \
  --exam-set-id f6adcc63-71dc-412c-9c8d-a4609df454ff \
  --variant-id 37e3d65f-e540-4e34-b438-549e731be3b0 \
  --output exam-a.pdf \
  --title "Midterm Exam A" \
  --instructions "Fill bubbles completely. Use a dark pencil."
```

### Generation Inputs

- `--questions`
  - required
  - comma-separated integers
  - each value must be between `2` and `5`
- `--exam-set-id`
  - required
  - encoded into the QR payload
- `--variant-id`
  - required
  - encoded into the QR payload
- `--output`
  - optional
  - defaults to `omr-sheet.pdf`
- `--title`
  - optional
- `--instructions`
  - optional
  - default is `Fill bubbles completely.`

### Question Layout Rules

- questions are numbered starting at `1`
- each column holds at most `10` questions
- each question has `2` to `5` bubbles
- option labels are `A` through `E`
- if the sheet needs more space, new pages are added

### QR Payload Format

The QR code contains JSON in this form:

```json
{
  "examSetId": "f6adcc63-71dc-412c-9c8d-a4609df454ff",
  "variantId": "37e3d65f-e540-4e34-b438-549e731be3b0"
}
```

## Grade a Filled Sheet

Single PDF:

```bash
uv run omr-grade filled-sheet.pdf
```

Folder of PDFs:

```bash
uv run omr-grade filled-sheets/
```

`omr-grade` prints JSON only. It does not write annotated PDFs.

## `omr-grade` Output

### Single PDF Output

Example:

```json
{
  "qr_data": {
    "examSetId": "f6adcc63-71dc-412c-9c8d-a4609df454ff",
    "variantId": "37e3d65f-e540-4e34-b438-549e731be3b0"
  },
  "student_id": "63620",
  "marked_answers": {
    "1": ["D"],
    "2": ["C"],
    "3": ["B", "C"],
    "4": ["C"],
    "5": ["A"],
    "6": ["E"]
  },
  "omr_error": ""
}
```

Field meanings:

- `qr_data`
  - decoded QR payload
  - usually an object with `examSetId` and `variantId`
  - may be `null` if QR decoding fails
- `student_id`
  - 5-digit student number
- `marked_answers`
  - object keyed by question number as strings
  - each value is a list of selected options
  - multiple answers are allowed
- `omr_error`
  - empty string on success

### `marked_answers` Schema

Example:

```json
{
  "1": ["B"],
  "2": ["B", "D"],
  "3": []
}
```

Rules:

- keys are question numbers as strings
- values are arrays of option labels
- multiple marked choices are supported
- unanswered detected rows produce an empty array

### Folder Output

When the input is a directory, the output is a JSON array:

```json
[
  {
    "source_pdf": "student-a.pdf",
    "qr_data": {
      "examSetId": "f6adcc63-71dc-412c-9c8d-a4609df454ff",
      "variantId": "37e3d65f-e540-4e34-b438-549e731be3b0"
    },
    "student_id": "63620",
    "marked_answers": {
      "1": ["D"]
    },
    "omr_error": ""
  },
  {
    "source_pdf": "student-b.pdf",
    "qr_data": null,
    "student_id": "",
    "marked_answers": {},
    "omr_error": "filled-sheets/student-b.pdf: Required alignment marker 'top_right' was not detected"
  }
]
```

Directory behavior:

- only `.pdf` files are processed
- grading continues even if one file fails
- failed files still get a JSON result entry
- failed entries have:
  - `qr_data: null`
  - `student_id: ""`
  - `marked_answers: {}`
  - `omr_error` populated

## Annotate a Filled Sheet

Single PDF:

```bash
uv run omr-annotate filled-sheet.pdf --output annotated.pdf
```

Directory:

```bash
uv run omr-annotate filled-sheets/ --output graded-pdfs/
```

With an answer key:

```bash
uv run omr-annotate filled-sheet.pdf \
  --output annotated.pdf \
  --correct-answers tests/answer-key.json
```

Inline answer-key JSON also works:

```bash
uv run omr-annotate filled-sheet.pdf \
  --output annotated.pdf \
  --correct-answers '{"1":["D"],"3":["B","C"]}'
```

## `omr-annotate` Output

Single-file example:

```json
{
  "qr_data": {
    "examSetId": "f6adcc63-71dc-412c-9c8d-a4609df454ff",
    "variantId": "37e3d65f-e540-4e34-b438-549e731be3b0"
  },
  "student_id": "63620",
  "marked_answers": {
    "1": ["D"]
  },
  "omr_error": "",
  "annotated_pdf": "/tmp/annotated.pdf"
}
```

Additional field:

- `annotated_pdf`
  - path of the written annotated PDF

### Annotation Behavior

The annotation overlay includes:

- `Student ID: ...`
- the raw `examSetId`
- the raw `variantId`
- red watermark text below the QR area
- `OMR Error: ...` for failed files
- optional faint red correct-answer overlays

Correct-answer overlays are only drawn for question rows actually detected on the sheet. Extra keys in the answer key are ignored for non-existent rows.

### Output Path Rules

For a single input PDF:

- if `--output` ends with `.pdf`, that exact file is written
- otherwise `--output` is treated as a directory and `<source-stem>-annotated.pdf` is created inside it

For a directory input:

- `--output` is treated as a directory
- each source PDF gets `<source-stem>-annotated.pdf`

## Correct Answer JSON Format

`--correct-answers` accepts either:

- a JSON string
- or a path to a JSON file

Supported format:

```json
{
  "1": ["D"],
  "2": ["B", "D"],
  "3": ["A"]
}
```

Also supported:

```json
{
  "answers": {
    "1": ["D"],
    "2": ["B", "D"]
  }
}
```

Rules:

- question keys may be strings or numbers
- values may be:
  - a string like `"D"`
  - or a list like `["B", "D"]`
- labels are normalized to uppercase
- labels should be `A` through `E`
- the answer key affects annotation only
- it does not change grading results

## Practical Examples

### Generate, Grade, Annotate One Sheet

```bash
uv run omr \
  --questions 4,4,5,3,2,5 \
  --exam-set-id exam-set-001 \
  --variant-id variant-a \
  --output sheet.pdf

uv run omr-grade sheet-filled.pdf

uv run omr-annotate \
  sheet-filled.pdf \
  --output sheet-filled-annotated.pdf \
  --correct-answers tests/answer-key.json
```

### Grade a Folder and Keep Going on Errors

```bash
uv run omr-grade scans/
```

If one file is bad, the command still returns JSON entries for the rest.

### Annotate a Folder for Manual Review

```bash
uv run omr-annotate scans/ --output reviewed/
```

### Pipe Grading Output to a File

```bash
uv run omr-grade scans/ > results.json
```

## Limitations

- grading requires the current marker-based sheet format
- grading currently reads only the first page of each PDF
- student ID is fixed to 5 digits
- option labels are limited to `A-E`
- registration is marker-first
- QR is used for payload extraction, not as the primary geometric anchor

## Testing

Run the full test suite:

```bash
uv run --with pytest pytest -q
```

Tests generate their PDF fixtures dynamically and clean them up automatically. No checked-in graded sample PDFs are required.
