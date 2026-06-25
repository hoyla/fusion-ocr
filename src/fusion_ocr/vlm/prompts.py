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
