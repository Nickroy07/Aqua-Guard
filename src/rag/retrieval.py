from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class GuidanceSnippet:
    title: str
    path: str
    snippet: str
    score: float


def _load_kb(path: Path) -> List[tuple[str, str]]:
    if not path.exists():
        return [("General Guidance", "Inspect fixtures and verify meter readings prior to taking action.")]
    text = path.read_text(encoding="utf-8")
    entries: List[tuple[str, str]] = []
    current_title = "General Guidance"
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                entries.append((current_title, " ".join(current_lines).strip()))
                current_lines = []
            current_title = line.replace("## ", "").strip()
        elif line and not line.startswith("# "):
            current_lines.append(line.strip())
    if current_lines:
        entries.append((current_title, " ".join(current_lines).strip()))
    return entries


def retrieve_guidance(query: str, kb_path: Path, top_k: int = 2) -> List[GuidanceSnippet]:
    entries = _load_kb(kb_path)
    if not entries:
        return []
    corpus = [f"{e[0]}: {e[1]}" for e in entries]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked_indices = scores.argsort()[::-1][:top_k]
    except Exception:
        ranked_indices = list(range(min(top_k, len(entries))))
        scores = [1.0] * len(ranked_indices)

    snippets: List[GuidanceSnippet] = []
    for idx in ranked_indices:
        title, snippet = entries[int(idx)]
        snippets.append(
            GuidanceSnippet(
                title=title,
                path=str(kb_path),
                snippet=snippet,
                score=float(scores[int(idx)]) if idx < len(scores) else 0.0,
            )
        )
    return snippets


def build_recommendation(query: str, kb_path: Path) -> str:
    snippets = retrieve_guidance(query, kb_path, top_k=2)
    if not snippets:
        return "No local guidance retrieved. Perform standard meter and fixture inspection."

    lead = f"Recommended Operational Action: {snippets[0].title}"
    evidence_lines = [
        f"- Evidence [{s.title}] ({Path(s.path).name}): {s.snippet}"
        for s in snippets
    ]
    return lead + "\n" + "\n".join(evidence_lines)
