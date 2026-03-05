#!/usr/bin/env python
"""
Benchmark Evaluator - Independent metric evaluation (no weighted merge)

Metrics:
1) Gap tracking (LLM, per tutor turn): mentioned vs newly resolved gaps (strict criteria)
2) Source faithfulness (LLM, per tutor turn): 1-5 score against source text
   - Evaluated only for gaps mentioned in metric 1 on that turn
3) Teaching quality (LLM, per tutor turn): insightfulness + applicability (1-5)
4) Turn count (non-LLM): student/tutor interaction counts

Supports:
- Single-session transcript: {"transcript": [...], "entry": {...}}
- Multi-session transcript: {"sessions": [{"transcript": [...], "entry": {...}, ...}, ...]}
"""

import json
import logging
from pathlib import Path

from benchmark.data_generation.llm_utils import call_llm_json, load_prompt

logger = logging.getLogger("benchmark.evaluation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECENT_CONTEXT_WINDOW = 8  # messages (student+tutor) to provide around each turn


def _format_profile(profile: dict) -> str:
    """Format student profile for prompt."""
    parts = []
    if profile.get("personality"):
        parts.append(f"Personality: {profile['personality']}")
    if profile.get("education_background"):
        parts.append(f"Background: {profile['education_background']}")
    if profile.get("learning_purpose"):
        parts.append(f"Purpose: {profile['learning_purpose']}")
    ks = profile.get("knowledge_state", {})
    if ks.get("known_well"):
        parts.append(f"Known well: {', '.join(ks['known_well'][:5])}")
    if ks.get("partially_known"):
        parts.append(f"Partially known: {', '.join(ks['partially_known'][:5])}")
    if ks.get("unknown"):
        parts.append(f"Unknown: {', '.join(ks['unknown'][:5])}")
    if profile.get("beliefs"):
        parts.append(f"Beliefs (may be misconceptions): {profile['beliefs']}")
    return "\n".join(parts) if parts else "(no profile)"


def _format_task(task: dict) -> str:
    """Format task for prompt."""
    parts = []
    if task.get("title"):
        parts.append(f"Title: {task['title']}")
    if task.get("description"):
        parts.append(f"Description: {task['description']}")
    if task.get("success_criteria"):
        parts.append(f"Success criteria: {task['success_criteria']}")
    if task.get("target_gaps"):
        parts.append(f"Target gaps: {task['target_gaps']}")
    return "\n".join(parts) if parts else "(no task)"


def _normalize_source_by_page(source_content: dict | None) -> dict[int, str]:
    """Normalize source_content keys to int (JSON may have str page keys)."""
    if not source_content:
        return {}
    out = {}
    for k, v in source_content.items():
        pk = int(k) if isinstance(k, str) and k.isdigit() else k
        if isinstance(pk, int):
            out[pk] = v or ""
    return out


def _format_transcript(transcript: list[dict]) -> str:
    """Format transcript for prompt."""
    lines = []
    for i, msg in enumerate(transcript, 1):
        role = msg.get("role", "?")
        content = (msg.get("content", "") or "")[:900]
        if len((msg.get("content") or "")) > 900:
            content += "..."
        lines.append(f"[{i}] {role.upper()}: {content}")
    return "\n\n".join(lines) if lines else "(empty)"


def _filter_dialog_messages(transcript: list[dict]) -> list[dict]:
    """Keep only student/tutor messages for evaluation scope."""
    return [m for m in transcript if m.get("role") in {"student", "tutor"}]


def _extract_turn_pairs(dialog_msgs: list[dict]) -> list[dict]:
    """
    Build per-turn student->tutor pairs.

    Returns list of:
      {
        "turn_index": int,
        "student_message": str,
        "tutor_response": str,
        "student_msg_index": int,   # index in dialog_msgs
      }
    """
    turns: list[dict] = []
    turn_index = 0
    for i in range(len(dialog_msgs) - 1):
        a, b = dialog_msgs[i], dialog_msgs[i + 1]
        if a.get("role") == "student" and b.get("role") == "tutor":
            turn_index += 1
            turns.append(
                {
                    "turn_index": turn_index,
                    "student_message": a.get("content", ""),
                    "tutor_response": b.get("content", ""),
                    "student_msg_index": i,
                }
            )
    return turns


def _get_recent_context(dialog_msgs: list[dict], student_msg_index: int, window: int = RECENT_CONTEXT_WINDOW) -> str:
    """
    Return recent context ending at current student message.

    This intentionally provides more than just current turn for metric-1 robustness.
    """
    start = max(0, student_msg_index - window + 1)
    snippet = dialog_msgs[start : student_msg_index + 1]
    return _format_transcript(snippet)


def _build_gap_map(gaps: list[dict]) -> dict[str, dict]:
    """Build gap_id -> gap dict mapping."""
    out = {}
    for g in gaps:
        gid = str(g.get("gap_id", "")).strip()
        if gid:
            out[gid] = g
    return out


def _format_gap_catalog(gaps: list[dict]) -> str:
    """Format full gap catalog for metric-1 prompt."""
    if not gaps:
        return "(no gaps)"
    lines = []
    for g in gaps:
        gid = g.get("gap_id", "unknown")
        concept = g.get("target_concept", "?")
        desc = (g.get("description", "") or "")[:350]
        mani = (g.get("manifestation", "") or "")[:240]
        corr = (g.get("correct_understanding", "") or "")[:320]
        lines.append(
            f"- gap_id: {gid}\n"
            f"  concept: {concept}\n"
            f"  description: {desc}\n"
            f"  manifestation: {mani}\n"
            f"  expected_correct_understanding: {corr}"
        )
    return "\n".join(lines)


def _format_mentioned_gaps_with_source(mentioned_gap_ids: list[str], gap_by_id: dict[str, dict], source_content: dict | None) -> str:
    """Format only mentioned gaps and their source excerpts for metric-2."""
    if not mentioned_gap_ids:
        return "(no mentioned gaps)"

    src_by_page = _normalize_source_by_page(source_content)
    lines = []
    for gid in mentioned_gap_ids:
        gap = gap_by_id.get(gid)
        if not gap:
            continue
        concept = gap.get("target_concept", "?")
        desc = (gap.get("description", "") or "")[:260]
        lines.append(f"### {gid} - {concept}")
        lines.append(f"Description: {desc}")
        pages = sorted(set(gap.get("source_pages", [])))
        if pages:
            lines.append(f"Source pages: {pages}")
        for p in pages:
            text = src_by_page.get(p, "")
            if text:
                excerpt = text[:1800] + ("..." if len(text) > 1800 else "")
                lines.append(f"Source page {p}:\n{excerpt}")
        lines.append("")
    return "\n".join(lines).strip() or "(no source excerpt for mentioned gaps)"


async def evaluate_gap_tracking_turn(
    *,
    entry: dict,
    turn_index: int,
    student_message: str,
    tutor_response: str,
    recent_context: str,
    previously_mentioned_gap_ids: list[str],
    previously_resolved_gap_ids: list[str],
    temperature: float,
) -> dict:
    """
    Metric-1 (LLM): detect mentioned gaps and newly resolved gaps on this tutor turn.

    Strict resolution is enforced in prompt:
      - "resolved" requires clear correction + concrete closure evidence, not a casual mention.
    """
    prompt_cfg = load_prompt("eval_gap_tracking_turn")
    profile = entry.get("profile", {})
    gaps = entry.get("gaps", [])
    task = entry.get("task", {})
    gap_by_id = _build_gap_map(gaps)
    valid_gap_ids = set(gap_by_id.keys())
    unresolved = valid_gap_ids - set(previously_resolved_gap_ids)

    user_prompt = prompt_cfg["user_template"].format(
        profile_summary=_format_profile(profile),
        task_summary=_format_task(task),
        gap_catalog=_format_gap_catalog(gaps),
        recent_context=recent_context,
        student_message=student_message,
        tutor_response=tutor_response,
        turn_index=turn_index,
        previously_mentioned_gap_ids=sorted(set(previously_mentioned_gap_ids)),
        previously_resolved_gap_ids=sorted(set(previously_resolved_gap_ids)),
    )

    try:
        result = await call_llm_json(
            user_prompt=user_prompt,
            system_prompt=prompt_cfg["system"],
            temperature=temperature,
            max_tokens=1024,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Gap tracking failed at turn %d: %s", turn_index, e)
        return {
            "turn_index": turn_index,
            "mentioned_gap_ids": [],
            "resolved_gap_ids_new": [],
            "rationale": f"Evaluation failed: {e}",
            "error": str(e),
        }

    mentioned = [gid for gid in result.get("mentioned_gap_ids", []) if gid in valid_gap_ids]
    mentioned = sorted(set(mentioned))

    resolved_new_raw = [gid for gid in result.get("resolved_gap_ids_new", []) if gid in unresolved]
    resolved_new = sorted(set(resolved_new_raw))
    # Resolved should be a subset of mentioned on this turn for consistency.
    resolved_new = [gid for gid in resolved_new if gid in mentioned]

    return {
        "turn_index": turn_index,
        "mentioned_gap_ids": mentioned,
        "resolved_gap_ids_new": resolved_new,
        "rationale": result.get("rationale", ""),
    }


async def evaluate_source_faithfulness_turn(
    *,
    turn_index: int,
    student_message: str,
    tutor_response: str,
    recent_context: str,
    mentioned_gap_ids: list[str],
    gap_by_id: dict[str, dict],
    source_content: dict | None,
    temperature: float,
) -> dict:
    """
    Metric-2 (LLM): faithfulness (1-5) against source text of mentioned gaps only.
    """
    if not mentioned_gap_ids:
        return {
            "turn_index": turn_index,
            "mentioned_gap_ids": [],
            "not_applicable": True,
            "reason": "No mentioned gaps on this turn.",
        }

    prompt_cfg = load_prompt("eval_source_faithfulness_turn")
    source_for_mentioned = _format_mentioned_gaps_with_source(mentioned_gap_ids, gap_by_id, source_content)
    if source_for_mentioned.startswith("(no source excerpt"):
        return {
            "turn_index": turn_index,
            "mentioned_gap_ids": mentioned_gap_ids,
            "not_applicable": True,
            "reason": "Mentioned gaps have no usable source excerpts.",
        }

    user_prompt = prompt_cfg["user_template"].format(
        turn_index=turn_index,
        recent_context=recent_context,
        student_message=student_message,
        tutor_response=tutor_response,
        mentioned_gap_ids=mentioned_gap_ids,
        source_content_for_mentioned_gaps=source_for_mentioned,
    )

    try:
        result = await call_llm_json(
            user_prompt=user_prompt,
            system_prompt=prompt_cfg["system"],
            temperature=temperature,
            max_tokens=900,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Source faithfulness failed at turn %d: %s", turn_index, e)
        return {
            "turn_index": turn_index,
            "mentioned_gap_ids": mentioned_gap_ids,
            "not_applicable": True,
            "reason": f"Evaluation failed: {e}",
            "error": str(e),
        }

    score = result.get("faithfulness_score")
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None
    if score is not None:
        score = max(1, min(5, score))

    return {
        "turn_index": turn_index,
        "mentioned_gap_ids": mentioned_gap_ids,
        "faithfulness_score": score,
        "rationale": result.get("rationale", ""),
        "contradictions": result.get("contradictions", result.get("unsupported_claims", [])),
        "not_applicable": score is None,
    }


def _build_source_faithfulness_summary(per_turn: list[dict]) -> dict:
    """Aggregate metric-2 stats: min/max/avg over scored turns only."""
    scored = [t["faithfulness_score"] for t in per_turn if not t.get("not_applicable") and t.get("faithfulness_score") is not None]
    if not scored:
        return {
            "scale": "1-5",
            "num_scored_turns": 0,
            "max_score": None,
            "min_score": None,
            "avg_score": None,
            "per_turn": per_turn,
        }
    return {
        "scale": "1-5",
        "num_scored_turns": len(scored),
        "max_score": max(scored),
        "min_score": min(scored),
        "avg_score": round(sum(scored) / len(scored), 2),
        "per_turn": per_turn,
    }


async def evaluate_teaching_quality_turn(
    *,
    turn_index: int,
    student_message: str,
    tutor_response: str,
    recent_context: str,
    temperature: float,
) -> dict:
    """
    Metric-3 (LLM): turn-level teaching quality on two dimensions:
      - insightfulness (1-5)
      - applicability (1-5)
    """
    prompt_cfg = load_prompt("eval_teaching_quality_turn")
    user_prompt = prompt_cfg["user_template"].format(
        turn_index=turn_index,
        recent_context=recent_context,
        student_message=student_message,
        tutor_response=tutor_response,
    )

    try:
        result = await call_llm_json(
            user_prompt=user_prompt,
            system_prompt=prompt_cfg["system"],
            temperature=temperature,
            max_tokens=700,
        )
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Teaching quality failed at turn %d: %s", turn_index, e)
        return {
            "turn_index": turn_index,
            "insightfulness_score": None,
            "applicability_score": None,
            "rationale": f"Evaluation failed: {e}",
            "not_applicable": True,
            "error": str(e),
        }

    insight = result.get("insightfulness_score")
    applicability = result.get("applicability_score")
    try:
        insight = int(insight) if insight is not None else None
    except (TypeError, ValueError):
        insight = None
    try:
        applicability = int(applicability) if applicability is not None else None
    except (TypeError, ValueError):
        applicability = None

    if insight is not None:
        insight = max(1, min(5, insight))
    if applicability is not None:
        applicability = max(1, min(5, applicability))

    return {
        "turn_index": turn_index,
        "insightfulness_score": insight,
        "applicability_score": applicability,
        "rationale": result.get("rationale", ""),
        "evidence": result.get("evidence", []),
        "not_applicable": insight is None and applicability is None,
    }


def _build_teaching_quality_summary(per_turn: list[dict]) -> dict:
    """Aggregate metric-3 stats over scored turns only."""
    insight_scores = [
        t.get("insightfulness_score")
        for t in per_turn
        if t.get("insightfulness_score") is not None and not t.get("not_applicable")
    ]
    applicability_scores = [
        t.get("applicability_score")
        for t in per_turn
        if t.get("applicability_score") is not None and not t.get("not_applicable")
    ]
    return {
        "scale": "1-5",
        "num_scored_turns_insightfulness": len(insight_scores),
        "num_scored_turns_applicability": len(applicability_scores),
        "avg_insightfulness": (
            round(sum(insight_scores) / len(insight_scores), 2)
            if insight_scores
            else None
        ),
        "avg_applicability": (
            round(sum(applicability_scores) / len(applicability_scores), 2)
            if applicability_scores
            else None
        ),
        "max_insightfulness": max(insight_scores) if insight_scores else None,
        "min_insightfulness": min(insight_scores) if insight_scores else None,
        "max_applicability": max(applicability_scores) if applicability_scores else None,
        "min_applicability": min(applicability_scores) if applicability_scores else None,
        "per_turn": per_turn,
    }


def _load_entry_by_id(entry_id: str) -> dict | None:
    """
    Try to load entry by entry_id from generated JSONL files.
    Supports nested benchmark_<timestamp>/_all_entries.jsonl layout.
    """
    generated_dir = PROJECT_ROOT / "benchmark" / "data" / "generated"
    candidates = sorted(generated_dir.glob("**/*.jsonl"))
    for jsonl_path in candidates:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("entry_id") == entry_id:
                    return entry
    return None


async def _evaluate_single_session(
    *,
    transcript: list[dict],
    entry: dict,
    entry_id: str,
    skip_turns: bool,
    temperature: float,
) -> dict:
    """Evaluate one session with 4 independent metrics."""
    dialog_msgs = _filter_dialog_messages(transcript)
    turns = _extract_turn_pairs(dialog_msgs)
    gaps = entry.get("gaps", [])
    gap_by_id = _build_gap_map(gaps)
    source_content = entry.get("source_content")

    logger.info(
        "Metrics enabled: gap_tracking(LLM), source_faithfulness(LLM)%s, teaching_quality(LLM), turn_count(non-LLM)",
        " [source present]" if source_content else " [source missing: may be N/A]",
    )

    student_turns = sum(1 for m in dialog_msgs if m.get("role") == "student")
    tutor_turns = sum(1 for m in dialog_msgs if m.get("role") == "tutor")
    turn_count_metric = {
        "student_turns": student_turns,
        "tutor_turns": tutor_turns,
        "paired_turns": len(turns),
    }

    gap_tracking_per_turn: list[dict] = []
    source_faithfulness_per_turn: list[dict] = []
    teaching_quality_per_turn: list[dict] = []
    mentioned_so_far: set[str] = set()
    resolved_so_far: set[str] = set()

    if not skip_turns:
        for t in turns:
            turn_index = t["turn_index"]
            student_message = t["student_message"]
            tutor_response = t["tutor_response"]
            recent_context = _get_recent_context(dialog_msgs, t["student_msg_index"])

            # Metric 1: gap mention + strict resolution
            gap_turn = await evaluate_gap_tracking_turn(
                entry=entry,
                turn_index=turn_index,
                student_message=student_message,
                tutor_response=tutor_response,
                recent_context=recent_context,
                previously_mentioned_gap_ids=sorted(mentioned_so_far),
                previously_resolved_gap_ids=sorted(resolved_so_far),
                temperature=temperature,
            )
            mentioned_turn = set(gap_turn.get("mentioned_gap_ids", []))
            resolved_turn_new = set(gap_turn.get("resolved_gap_ids_new", []))

            mentioned_so_far |= mentioned_turn
            resolved_so_far |= resolved_turn_new

            gap_turn["resolved_gap_ids_total"] = sorted(resolved_so_far)
            gap_turn["mentioned_count_turn"] = len(mentioned_turn)
            gap_turn["resolved_count_turn_new"] = len(resolved_turn_new)
            gap_turn["resolved_count_total"] = len(resolved_so_far)
            gap_tracking_per_turn.append(gap_turn)

            # Metric 2: source faithfulness using only mentioned gaps this turn
            faith_turn = await evaluate_source_faithfulness_turn(
                turn_index=turn_index,
                student_message=student_message,
                tutor_response=tutor_response,
                recent_context=recent_context,
                mentioned_gap_ids=sorted(mentioned_turn),
                gap_by_id=gap_by_id,
                source_content=source_content,
                temperature=temperature,
            )
            source_faithfulness_per_turn.append(faith_turn)

            # Metric 3: turn-level teaching quality
            quality_turn = await evaluate_teaching_quality_turn(
                turn_index=turn_index,
                student_message=student_message,
                tutor_response=tutor_response,
                recent_context=recent_context,
                temperature=temperature,
            )
            teaching_quality_per_turn.append(quality_turn)

            logger.info(
                "Turn %d: mentioned=%d, newly_resolved=%d, resolved_total=%d, faithfulness=%s, insightfulness=%s, applicability=%s",
                turn_index,
                len(mentioned_turn),
                len(resolved_turn_new),
                len(resolved_so_far),
                faith_turn.get("faithfulness_score", "N/A"),
                quality_turn.get("insightfulness_score", "N/A"),
                quality_turn.get("applicability_score", "N/A"),
            )

    gap_tracking_metric = {
        "total_gaps": len(gap_by_id),
        "mentioned_gap_ids_final": sorted(mentioned_so_far),
        "resolved_gap_ids_final": sorted(resolved_so_far),
        "resolved_gaps_final_count": len(resolved_so_far),
        "per_turn": gap_tracking_per_turn,
    }
    source_faithfulness_metric = _build_source_faithfulness_summary(source_faithfulness_per_turn)
    teaching_quality_metric = _build_teaching_quality_summary(teaching_quality_per_turn)

    return {
        "entry_id": entry_id,
        "actual_turns": len(turns),
        "metrics": {
            "gap_tracking": gap_tracking_metric,
            "source_faithfulness": source_faithfulness_metric,
            "teaching_quality": teaching_quality_metric,
            "turn_count": turn_count_metric,
        },
    }


def _aggregate_multi_session(session_results: list[dict]) -> dict:
    """Build lightweight aggregate view; each session gap stats stay independent."""
    if not session_results:
        return {}

    total_student_turns = 0
    total_tutor_turns = 0
    total_paired_turns = 0
    faith_scores = []
    insight_scores = []
    applicability_scores = []
    resolved_counts = []
    total_gaps_counts = []

    for s in session_results:
        tc = s.get("metrics", {}).get("turn_count", {})
        total_student_turns += tc.get("student_turns", 0)
        total_tutor_turns += tc.get("tutor_turns", 0)
        total_paired_turns += tc.get("paired_turns", 0)

        gt = s.get("metrics", {}).get("gap_tracking", {})
        resolved_counts.append(gt.get("resolved_gaps_final_count", 0))
        total_gaps_counts.append(gt.get("total_gaps", 0))

        sf = s.get("metrics", {}).get("source_faithfulness", {})
        for t in sf.get("per_turn", []):
            score = t.get("faithfulness_score")
            if score is not None and not t.get("not_applicable"):
                faith_scores.append(score)

        tq = s.get("metrics", {}).get("teaching_quality", {})
        for t in tq.get("per_turn", []):
            insight = t.get("insightfulness_score")
            applicability = t.get("applicability_score")
            if insight is not None and not t.get("not_applicable"):
                insight_scores.append(insight)
            if applicability is not None and not t.get("not_applicable"):
                applicability_scores.append(applicability)

    return {
        "turn_count": {
            "student_turns_total": total_student_turns,
            "tutor_turns_total": total_tutor_turns,
            "paired_turns_total": total_paired_turns,
        },
        "gap_tracking": {
            "resolved_gaps_per_session": resolved_counts,
            "total_gaps_per_session": total_gaps_counts,
        },
        "source_faithfulness": {
            "scale": "1-5",
            "num_scored_turns_total": len(faith_scores),
            "max_score_overall": max(faith_scores) if faith_scores else None,
            "min_score_overall": min(faith_scores) if faith_scores else None,
            "avg_score_overall": round(sum(faith_scores) / len(faith_scores), 2) if faith_scores else None,
        },
        "teaching_quality": {
            "scale": "1-5",
            "num_scored_turns_insightfulness_total": len(insight_scores),
            "num_scored_turns_applicability_total": len(applicability_scores),
            "avg_insightfulness_overall": (
                round(sum(insight_scores) / len(insight_scores), 2)
                if insight_scores
                else None
            ),
            "avg_applicability_overall": (
                round(sum(applicability_scores) / len(applicability_scores), 2)
                if applicability_scores
                else None
            ),
        },
    }


async def evaluate_transcript(
    transcript_path: str | Path,
    skip_turns: bool = False,
    temperature: float = 0.2,
) -> dict:
    """
    Evaluate transcript using independent metrics (no weighted merge).

    Args:
        transcript_path: Path to transcript JSON
        skip_turns: If True, skip LLM per-turn metrics and keep only turn_count
        temperature: LLM temperature for metric-1 and metric-2
    """
    path = Path(transcript_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Multi-session format
    if "sessions" in data:
        sessions = data["sessions"]
        profile_id = data.get("profile_id", "unknown")
        session_results = []

        for i, sess in enumerate(sessions):
            transcript = sess.get("transcript", [])
            entry = sess.get("entry")
            entry_id = sess.get("entry_id", f"session_{i+1}")

            if not entry:
                entry = _load_entry_by_id(entry_id)
            if not entry:
                logger.warning(
                    "Session %s has no entry; skipping (run multi-session again to save entry)",
                    entry_id,
                )
                continue

            logger.info("Evaluating session %d/%d: %s", i + 1, len(sessions), entry_id)
            sess_result = await _evaluate_single_session(
                transcript=transcript,
                entry=entry,
                entry_id=entry_id,
                skip_turns=skip_turns,
                temperature=temperature,
            )
            session_results.append(sess_result)

        if not session_results:
            raise ValueError(
                "No sessions could be evaluated (missing 'entry' in each session; "
                "re-run multi-session to save entries)"
            )

        return {
            "profile_id": profile_id,
            "transcript_path": str(path),
            "num_sessions": len(session_results),
            "sessions": session_results,
            "aggregate": _aggregate_multi_session(session_results),
        }

    # Single-session format
    transcript = data.get("transcript", [])
    entry = data.get("entry", {})
    if not entry:
        raise ValueError("Transcript must contain 'entry' (benchmark entry with profile, gaps, task)")

    result = await _evaluate_single_session(
        transcript=transcript,
        entry=entry,
        entry_id=data.get("entry_id", "unknown"),
        skip_turns=skip_turns,
        temperature=temperature,
    )
    result["transcript_path"] = str(path)
    return result
