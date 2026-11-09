from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from .models import ReviewIssue


@dataclass(slots=True)
class ScriptBlock:
    label: str
    start: float
    end: float
    text: str

    @property
    def timecode(self) -> str:
        return f"{_format_time(self.start)}–{_format_time(self.end)}"


@dataclass(slots=True)
class Requirement:
    number: int
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


@dataclass(slots=True)
class Anomaly:
    requirement: Requirement
    source: ScriptBlock
    target_index: int | None
    preservation_score: float
    reason: str


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    relevance_floor: float = 0.06
    candidate_ratio: float = 0.80
    max_candidates_per_rule: int = 2
    minimum_time_overlap: float = 0.15
    preservation_floor: float = 0.60
    critical_confidence: float = 0.75


BLOCK_RE = re.compile(
    r"\*\*(?P<label>[^*]+)\*\*:?\s*\n"
    r"Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)\s*\n"
    r"(?P<text>.*?)(?=\n\n\*\*|\n--- SCRIPT END|$)",
    re.S,
)
RULE_RE = re.compile(r"^\s*(?P<number>\d+)\.\s*(?P<title>[^:]+):\s*(?P<text>.+?)\s*$")


def _seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(parts[0])


def _format_time(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0), 60)
    return f"{int(minutes):02d}:{remainder:05.2f}"


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "issue"


def parse_blocks(screenplay: str) -> list[ScriptBlock]:
    return [
        ScriptBlock(
            label=_clean(match.group("label")),
            start=_seconds(match.group("start")),
            end=_seconds(match.group("end")),
            text=_clean(match.group("text")),
        )
        for match in BLOCK_RE.finditer(screenplay)
    ]


def parse_requirements(rules: str) -> list[Requirement]:
    parsed: list[Requirement] = []
    for line in rules.splitlines():
        match = RULE_RE.match(line)
        if match:
            parsed.append(
                Requirement(
                    number=int(match.group("number")),
                    title=_clean(match.group("title")),
                    text=_clean(match.group("text")),
                )
            )
    return parsed


def character_pairs(character_map: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for character in character_map.get("characters", []):
        target = _clean(str(character.get("adapted_name", "")))
        if not target:
            continue
        for field in ("source_name", "romanized_name"):
            source = _clean(str(character.get(field, "")))
            if source and source.casefold() != target.casefold():
                pairs.append((source, target))
    return pairs


def normalize_source_text(text: str, pairs: Iterable[tuple[str, str]]) -> str:
    normalized = text
    for source, target in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(re.escape(source), target, normalized, flags=re.I)
    return normalized


def _block_for_term(blocks: list[ScriptBlock], term: str) -> ScriptBlock | None:
    return next((block for block in blocks if re.search(re.escape(term), block.text, re.I)), None)


def _scene_timecode(screenplay: str) -> str:
    start = re.search(r"^Scene start:\s*([0-9:.]+)", screenplay, re.I | re.M)
    end = re.search(r"^Scene end:\s*([0-9:.]+)", screenplay, re.I | re.M)
    return f"{start.group(1)}–{end.group(1)}" if start and end else "unknown"


def _extract_forbidden_terms(requirement: Requirement) -> list[str]:
    instruction = requirement.text
    if not re.search(r"\b(?:do not retain|must not contain|forbid|avoid using|remove the term)\b", instruction, re.I):
        return []
    quoted = re.findall(r"[\"“]([^\"”]{2,60})[\"”]", instruction)
    if quoted:
        return [_clean(item) for item in quoted]
    match = re.search(r"(?:term|word|phrase)\s+([A-Za-z][A-Za-z0-9_-]{1,40})", instruction, re.I)
    return [match.group(1)] if match else []


def _direct_constraint_issues(
    script: str,
    adapted_blocks: list[ScriptBlock],
    pairs: list[tuple[str, str]],
    requirements: list[Requirement],
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []

    for source, target in pairs:
        if not re.search(re.escape(source), script, re.I):
            continue
        block = _block_for_term(adapted_blocks, source)
        issues.append(
            ReviewIssue(
                id=f"name-{_slug(source)}",
                severity="warning",
                category="Character-map mismatch",
                title="A source character name remains in the adaptation",
                explanation="This flag was generated by comparing the draft with the supplied character map.",
                evidence=f"Character map: {source} → {target}",
                timecode=block.timecode if block else "unknown",
                proposal=f"Replace “{source}” with “{target}”.",
                confidence=1.0,
                detected_by="input-derived constraint",
            )
        )

    for requirement in requirements:
        for term in _extract_forbidden_terms(requirement):
            if not re.search(re.escape(term), script, re.I):
                continue
            block = _block_for_term(adapted_blocks, term)
            issues.append(
                ReviewIssue(
                    id=f"rule-{requirement.number}-{_slug(term)}",
                    severity="warning",
                    category="Forbidden-term violation",
                    title="The draft retains a term forbidden by the supplied rules",
                    explanation=f"The model derived this constraint from requirement {requirement.number}, not from a built-in term list.",
                    evidence=requirement.full_text,
                    timecode=block.timecode if block else _scene_timecode(script),
                    proposal=f"Replace “{term}” with the target-culture equivalent specified by requirement {requirement.number}.",
                    confidence=1.0,
                    detected_by="input-derived constraint",
                )
            )
    return issues


def _is_preservation_requirement(requirement: Requirement) -> bool:
    if _extract_forbidden_terms(requirement):
        return False
    return bool(
        re.search(
            r"\b(?:preserve|must|required|required to|do not (?:soften|remove|omit)|protect|retain)\b",
            requirement.text,
            re.I,
        )
    )


def _feature_model(documents: list[str]):
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    lowercase=True,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
        ]
    ).fit_transform(documents)


def _overlap(left: ScriptBlock, right: ScriptBlock) -> float:
    intersection = max(0.0, min(left.end, right.end) - max(left.start, right.start))
    denominator = max(min(left.end - left.start, right.end - right.start), 0.01)
    return intersection / denominator


def _semantic_anomalies(
    source_blocks: list[ScriptBlock],
    adapted_blocks: list[ScriptBlock],
    requirements: list[Requirement],
    config: AlignmentConfig,
) -> tuple[list[Anomaly], dict]:
    preservation_rules = [item for item in requirements if _is_preservation_requirement(item)]
    if not source_blocks or not adapted_blocks or not preservation_rules:
        return [], {"documents": 0, "features": 0}

    documents = (
        [item.text for item in source_blocks]
        + [item.text for item in adapted_blocks]
        + [item.full_text for item in preservation_rules]
    )
    matrix = _feature_model(documents)
    similarities = cosine_similarity(matrix)
    source_count = len(source_blocks)
    adapted_count = len(adapted_blocks)
    anomalies: list[Anomaly] = []

    for rule_offset, requirement in enumerate(preservation_rules):
        rule_index = source_count + adapted_count + rule_offset
        relevance = similarities[:source_count, rule_index]
        top_relevance = float(np.max(relevance))
        if top_relevance < config.relevance_floor:
            continue

        candidate_floor = max(config.relevance_floor, top_relevance * config.candidate_ratio)
        candidates = np.flatnonzero(relevance >= candidate_floor)
        candidates = sorted(candidates, key=lambda index: relevance[index], reverse=True)[
            : config.max_candidates_per_rule
        ]

        for source_index in candidates:
            source = source_blocks[int(source_index)]
            overlapping = [
                index
                for index, target in enumerate(adapted_blocks)
                if _overlap(source, target) >= config.minimum_time_overlap
            ]
            if not overlapping:
                anomalies.append(
                    Anomaly(
                        requirement=requirement,
                        source=source,
                        target_index=None,
                        preservation_score=0.0,
                        reason="missing",
                    )
                )
                continue

            target_index = max(
                overlapping,
                key=lambda index: similarities[int(source_index), source_count + index],
            )
            source_to_target = float(similarities[int(source_index), source_count + target_index])
            target_to_rule = float(similarities[source_count + target_index, rule_index])
            relative_rule_coverage = min(target_to_rule / max(float(relevance[source_index]), 0.001), 1.0)
            preservation_score = max(source_to_target, relative_rule_coverage)
            if preservation_score < config.preservation_floor:
                anomalies.append(
                    Anomaly(
                        requirement=requirement,
                        source=source,
                        target_index=target_index,
                        preservation_score=preservation_score,
                        reason="semantic drift",
                    )
                )

    return anomalies, {"documents": len(documents), "features": int(matrix.shape[1])}


def _group_anomalies(anomalies: list[Anomaly]) -> list[list[Anomaly]]:
    groups: list[list[Anomaly]] = []
    for anomaly in sorted(anomalies, key=lambda item: (item.source.start, item.requirement.number)):
        destination = None
        for group in groups:
            same_target = anomaly.target_index is not None and anomaly.target_index == group[0].target_index
            overlapping_source = any(
                min(anomaly.source.end, item.source.end) > max(anomaly.source.start, item.source.start)
                for item in group
            )
            if same_target or overlapping_source:
                destination = group
                break
        if destination is None:
            groups.append([anomaly])
        else:
            destination.append(anomaly)
    return groups


def _anomaly_issue(group: list[Anomaly], config: AlignmentConfig) -> ReviewIssue:
    start = min(item.source.start for item in group)
    end = max(item.source.end for item in group)
    rule_numbers = sorted({item.requirement.number for item in group})
    source_beats = list(dict.fromkeys(item.source.text for item in group))
    missing = all(item.reason == "missing" for item in group)
    confidence = 1.0 - max(item.preservation_score for item in group)
    rules_label = ", ".join(str(number) for number in rule_numbers)
    evidence = " ".join(source_beats)
    return ReviewIssue(
        id=f"alignment-rules-{'-'.join(map(str, rule_numbers))}-{int(start * 100)}",
        severity="critical" if confidence >= config.critical_confidence else "warning",
        category="Missing required beat" if missing else "Semantic drift",
        title="A rule-protected source beat is missing" if missing else "The adaptation changes a rule-protected meaning",
        explanation=(
            f"The alignment model linked this source evidence to requirement(s) {rules_label} and found "
            f"{'no overlapping adapted block' if missing else 'low semantic preservation'} (score {1 - confidence:.2f})."
        ),
        evidence=evidence,
        timecode=f"{_format_time(start)}–{_format_time(end)}",
        proposal=f"Restore the meaning of this evidence while applying requirement(s) {rules_label}: {evidence}",
        confidence=round(min(max(confidence, 0.0), 1.0), 3),
        detected_by="TF-IDF + timestamp alignment model",
    )


def alignment_review(
    script: str,
    character_map: dict,
    rules: str,
    evidence: str,
    config: AlignmentConfig | None = None,
) -> tuple[list[ReviewIssue], dict]:
    """Fit a fresh alignment model from the four supplied inputs.

    No show title, character name, venue, plot event, issue count or expected flaw is
    encoded here. Exact constraints come from the map/rules; semantic anomalies come
    from source/adaptation similarity and timestamp alignment.
    """
    config = config or AlignmentConfig()
    pairs = character_pairs(character_map)
    source_blocks = parse_blocks(evidence)
    adapted_blocks = parse_blocks(script)
    requirements = parse_requirements(rules)
    normalized_source = [
        ScriptBlock(
            label=block.label,
            start=block.start,
            end=block.end,
            text=normalize_source_text(block.text, pairs),
        )
        for block in source_blocks
    ]

    direct = _direct_constraint_issues(script, adapted_blocks, pairs, requirements)
    anomalies, model_stats = _semantic_anomalies(
        normalized_source, adapted_blocks, requirements, config
    )
    learned = [_anomaly_issue(group, config) for group in _group_anomalies(anomalies)]
    issues = direct + learned
    issues.sort(key=lambda item: (_seconds(item.timecode.split("–", 1)[0]) if item.timecode != "unknown" else 10**9, item.id))

    metadata = {
        "model": "per-run TF-IDF word/character feature union + timestamp alignment",
        "source_blocks": len(source_blocks),
        "adapted_blocks": len(adapted_blocks),
        "requirements": len(requirements),
        "documents_fitted": model_stats["documents"],
        "feature_count": model_stats["features"],
        "hardcoded_show_knowledge": False,
        "thresholds": {
            "relevance_floor": config.relevance_floor,
            "candidate_ratio": config.candidate_ratio,
            "max_candidates_per_rule": config.max_candidates_per_rule,
            "minimum_time_overlap": config.minimum_time_overlap,
            "preservation_floor": config.preservation_floor,
            "critical_confidence": config.critical_confidence,
        },
    }
    return issues, metadata


def merge_model_issues(alignment_issues: list[ReviewIssue], model_items: list[dict]) -> list[ReviewIssue]:
    merged = {item.id: item for item in alignment_issues}
    for index, raw in enumerate(model_items, start=1):
        try:
            confidence = float(raw.get("confidence", 0.65))
        except (TypeError, ValueError):
            confidence = 0.65
        if confidence < 0.6:
            continue
        issue_id = str(raw.get("id") or f"language-model-{index}").strip().lower().replace(" ", "-")
        if issue_id in merged:
            continue
        merged[issue_id] = ReviewIssue(
            id=issue_id,
            severity=str(raw.get("severity", "warning")),
            category=str(raw.get("category", "Semantic mismatch")),
            title=str(raw.get("title", "Possible adaptation mismatch")),
            explanation=str(raw.get("explanation", "The configured model found a possible source-to-adaptation mismatch.")),
            evidence=str(raw.get("evidence", "Model-linked source evidence")),
            timecode=str(raw.get("timecode", "unknown")),
            proposal=str(raw.get("proposal", "Review this beat against the source.")),
            confidence=min(max(confidence, 0.0), 1.0),
            detected_by="configured language model",
        )
    return list(merged.values())
