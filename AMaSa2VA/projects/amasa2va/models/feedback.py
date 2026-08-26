"""Stable sample matching for Video-MME feedback routing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


_OPTION = re.compile(r"^[A-D][.)]\s+", re.IGNORECASE)
_FRAME = re.compile(r"\bFrame-?\d+\b", re.IGNORECASE)
_QUESTION_LABEL = re.compile(r"\bQuestion\s*:\s*", re.IGNORECASE)
_SPACE = re.compile(r"\s+")

_BOILERPLATE = (
    "These are the frames of a video.",
    "Select the best answer to the following multiple-choice question based on the video.",
    "Respond with only the letter (A, B, C, or D) of the correct option.",
    "This video's subtitles are listed below:",
)

_SUFFIXES = (
    "Answer with the option's letter from the given choices directly.",
    "Answer with the option's letter.",
    "Please directly answer with the option letter.",
    "请直接回答选项字母。",
    "Answer:",
)


def canonical_question(text: str) -> str:
    """Extract a prompt-independent question string for stable matching."""
    value = _FRAME.sub(" ", str(text or ""))
    matches = list(_QUESTION_LABEL.finditer(value))
    if matches:
        value = value[matches[-1].end():]
    for fixed in _BOILERPLATE:
        value = value.replace(fixed, " ")
    for suffix in _SUFFIXES:
        value = value.split(suffix, 1)[0]

    before_options = []
    for line in value.splitlines():
        line = line.strip()
        if _OPTION.match(line):
            break
        if line:
            before_options.append(line)

    # Video-MME questions are one line. Taking the final non-empty line also
    # discards optional subtitle text that may precede the question.
    question = before_options[-1] if before_options else value
    return _SPACE.sub(" ", question).strip().casefold()


def question_fingerprint(text: str) -> str:
    canonical = canonical_question(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def message_fingerprint(message: Iterable[dict[str, Any]]) -> str:
    text = "\n".join(
        str(item.get("value", ""))
        for item in message
        if item.get("type") == "text"
    )
    return question_fingerprint(text)


@dataclass
class QADenylist:
    indices: set[int] = field(default_factory=set)
    question_hashes: set[str] = field(default_factory=set)

    def __len__(self) -> int:
        return max(len(self.indices), len(self.question_hashes))

    def matches(
        self,
        message: Iterable[dict[str, Any]],
        sample_index: Optional[int] = None,
    ) -> bool:
        if sample_index is not None:
            try:
                if int(sample_index) in self.indices:
                    return True
            except (TypeError, ValueError):
                pass
        return message_fingerprint(message) in self.question_hashes


def load_qa_denylist(path: str) -> Optional[QADenylist]:
    """Load legacy integer lines or index/hash tab-separated records."""
    if not path:
        return None
    denylist = QADenylist()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for field_value in line.split("\t"):
            field_value = field_value.strip()
            if field_value.startswith("index:"):
                field_value = field_value.split(":", 1)[1]
            if field_value.startswith("question_sha256:"):
                digest = field_value.split(":", 1)[1].lower()
                if re.fullmatch(r"[0-9a-f]{64}", digest):
                    denylist.question_hashes.add(digest)
                continue
            try:
                denylist.indices.add(int(field_value))
            except ValueError:
                continue
    return denylist if denylist.indices or denylist.question_hashes else None
