"""VLM prompt templates. Kept separate so they can be tuned without touching the
client or the stage. Anti-hallucination instructions live here too — the fusion
ink-gate is the hard backstop, but a disciplined prompt reduces the work."""

TRANSCRIBE = (
    "Transcribe the text in this image exactly as written. Do not infer, complete, "
    "correct, or translate. If a portion is illegible, output [illegible] rather "
    "than guessing. Preserve line breaks."
)

TABLE = (
    "This image is a table. Reproduce it as GitHub-flavored Markdown, preserving the "
    "row and column structure and every cell's text. Do not invent values; use an "
    "empty cell where the source is blank or unreadable."
)

TRANSLATE = (
    "Translate the following document text into {target_lang}. Preserve structure "
    "(headings, lists, tables). Translate only — do not add commentary."
)

# Typhoon OCR is fine-tuned with its own instruction format; a generic transcribe
# prompt makes it echo its template instead of reading. This is the literal-text
# variant of its prompt — tables/page-numbers/checkboxes preserved, but the
# figure-DESCRIPTION/chart-ANALYSIS rules dropped (that's inference, not OCR, and
# would break defensibility). Verified on the Thai form.
TYPHOON_OCR = (
    "Extract all text from the image.\n\n"
    "Instructions:\n"
    "- Only return the clean Markdown.\n"
    "- Do not include any explanation or extra text.\n"
    "- You must include all information on the page.\n\n"
    "Formatting Rules:\n"
    "- Tables: Render tables using <table>...</table> in clean HTML format.\n"
    "- Page Numbers: Wrap page numbers in <page_number>...</page_number>.\n"
    "- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."
)


def select_prompt(model: str) -> str:
    """Pick the transcription prompt that matches the reader model. Specialist OCR
    models trained on a specific instruction need it; generalists take TRANSCRIBE."""
    if "typhoon" in (model or "").lower():
        return TYPHOON_OCR
    return TRANSCRIBE
