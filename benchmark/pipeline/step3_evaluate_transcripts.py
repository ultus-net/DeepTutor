#!/usr/bin/env python3
"""
Step 3: Evaluate transcripts for specified KBs.

Only evaluates transcripts that belong to input KB names.
Missing transcripts are recorded as errors (continue-on-error).

Input:
  Expected profile set from:
    <output_root>/entries/<kb_name>/profiles/<profile_id>/entries.jsonl
  Transcript:
    <output_root>/transcripts/<kb_name>/<backend>/<profile_id>.json

Output:
  <output_root>/evaluations/<kb_name>/<backend>/<profile_id>_eval.json

Manifest:
  <output_root>/manifests/step3_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluation.evaluator import evaluate_transcript

logger = logging.getLogger("benchmark.pipeline.step3")

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark" / "data" / "bench_pipeline"


def _parse_names(raw: str) -> list[str]:
    return sorted(set(n.strip() for n in raw.split(",") if n.strip()))


async def _evaluate_one_transcript(
    *,
    kb_name: str,
    profile_id: str,
    backend: str,
    transcript_path: Path,
    output_root: Path,
    temperature: float,
    skip_turns: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        out_path = output_root / "evaluations" / kb_name / backend / f"{profile_id}_eval.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kb_name": kb_name,
            "profile_id": profile_id,
            "backend": backend,
            "transcript_path": str(transcript_path),
            "evaluation_path": str(out_path),
            "status": "ok",
            "error": None,
        }
        try:
            result = await evaluate_transcript(
                transcript_path=transcript_path,
                skip_turns=skip_turns,
                temperature=temperature,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)
            logger.error("[%s/%s] %s eval failed: %s", kb_name, backend, profile_id, e)
        return record


def _extract_eval_summary(eval_data: dict) -> dict:
    agg = eval_data.get("aggregate", {})
    return {
        "turn_count": agg.get("turn_count", {}),
        "gap_tracking": agg.get("gap_tracking", {}),
        "source_faithfulness": agg.get("source_faithfulness", {}),
        "teaching_quality": agg.get("teaching_quality", {}),
    }


def _build_aggregate_summary(results: list[dict], output_root: Path) -> dict:
    grouped: dict[str, dict[str, Any]] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        backend = r.get("backend", "unknown")
        grouped.setdefault(
            backend,
            {
                "num_profiles": 0,
                "paired_turns_total": 0,
                "faithfulness_scores": [],
                "insightfulness_scores": [],
                "applicability_scores": [],
            },
        )
        try:
            with open(r["evaluation_path"], encoding="utf-8") as f:
                eval_data = json.load(f)
        except Exception:
            continue
        s = _extract_eval_summary(eval_data)
        grouped[backend]["num_profiles"] += 1
        grouped[backend]["paired_turns_total"] += (
            s.get("turn_count", {}).get("paired_turns_total", 0)
        )
        faith = s.get("source_faithfulness", {}).get("avg_score_overall")
        if isinstance(faith, (int, float)):
            grouped[backend]["faithfulness_scores"].append(float(faith))
        insight = s.get("teaching_quality", {}).get("avg_insightfulness_overall")
        if isinstance(insight, (int, float)):
            grouped[backend]["insightfulness_scores"].append(float(insight))
        app = s.get("teaching_quality", {}).get("avg_applicability_overall")
        if isinstance(app, (int, float)):
            grouped[backend]["applicability_scores"].append(float(app))

    out: dict[str, dict] = {}
    for backend, s in grouped.items():
        out[backend] = {
            "num_profiles": s["num_profiles"],
            "paired_turns_total": s["paired_turns_total"],
            "avg_faithfulness": (
                round(sum(s["faithfulness_scores"]) / len(s["faithfulness_scores"]), 2)
                if s["faithfulness_scores"]
                else None
            ),
            "avg_insightfulness": (
                round(sum(s["insightfulness_scores"]) / len(s["insightfulness_scores"]), 2)
                if s["insightfulness_scores"]
                else None
            ),
            "avg_applicability": (
                round(sum(s["applicability_scores"]) / len(s["applicability_scores"]), 2)
                if s["applicability_scores"]
                else None
            ),
        }
    return {
        "timestamp": datetime.now().isoformat(),
        "output_root": str(output_root),
        "by_backend": out,
    }


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Step3: evaluate transcripts")
    parser.add_argument("--kb-names", required=True, help="Comma-separated KB names")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Pipeline output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--backends",
        default="mock,deep_tutor",
        help="Comma-separated backends to evaluate",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="Max parallel transcript evaluations",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature for evaluation",
    )
    parser.add_argument(
        "--skip-turns",
        action="store_true",
        help="Skip per-turn LLM metrics and keep only turn_count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for compatibility; evaluations are overwritten by default.",
    )
    args = parser.parse_args()

    kb_names = _parse_names(args.kb_names)
    backends = _parse_names(args.backends)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()

    entries_root = output_root / "entries"
    transcripts_root = output_root / "transcripts"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    print(f"KBs: {len(kb_names)} | Backends: {backends}")
    print(f"Concurrency(transcript): {args.concurrency}")
    print(f"Output root: {output_root}")

    missing_errors: list[dict] = []
    tasks = []
    sem = asyncio.Semaphore(args.concurrency)

    for kb_name in kb_names:
        profiles_root = entries_root / kb_name / "profiles"
        if not profiles_root.exists():
            missing_errors.append(
                {
                    "kb_name": kb_name,
                    "profile_id": None,
                    "backend": None,
                    "error": f"Missing entries root for KB: {profiles_root}",
                }
            )
            logger.error("[%s] missing entries root: %s", kb_name, profiles_root)
            continue

        profile_ids = sorted(p.name for p in profiles_root.iterdir() if p.is_dir())
        for profile_id in profile_ids:
            for backend in backends:
                transcript_path = transcripts_root / kb_name / backend / f"{profile_id}.json"
                if not transcript_path.exists():
                    missing_errors.append(
                        {
                            "kb_name": kb_name,
                            "profile_id": profile_id,
                            "backend": backend,
                            "error": f"Missing transcript: {transcript_path}",
                        }
                    )
                    logger.error(
                        "[%s/%s] %s missing transcript", kb_name, backend, profile_id
                    )
                    continue
                tasks.append(
                    _evaluate_one_transcript(
                        kb_name=kb_name,
                        profile_id=profile_id,
                        backend=backend,
                        transcript_path=transcript_path,
                        output_root=output_root,
                        temperature=args.temperature,
                        skip_turns=args.skip_turns,
                        semaphore=sem,
                    )
                )

    logger.info("Launching %d transcript evaluation tasks", len(tasks))
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    eval_results = []
    task_errors = 0
    for r in task_results:
        if isinstance(r, Exception):
            task_errors += 1
            eval_results.append({"status": "error", "error": str(r)})
        else:
            eval_results.append(r)
            if r.get("status") != "ok":
                task_errors += 1

    aggregate = _build_aggregate_summary(eval_results, output_root)
    summary_path = manifests_root / "step3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    manifest = {
        "step": "step3_evaluate_transcripts",
        "timestamp": datetime.now().isoformat(),
        "kb_names": kb_names,
        "backends": backends,
        "output_root": str(output_root),
        "concurrency_transcript": args.concurrency,
        "overwrite": True,
        "missing_errors": missing_errors,
        "results": eval_results,
        "num_evaluated": len(eval_results),
        "num_errors": len(missing_errors) + task_errors,
        "summary_path": str(summary_path),
    }
    manifest_path = manifests_root / "step3_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSummary: {summary_path}")
    print(f"Manifest: {manifest_path}")
    print(
        f"Done. Evaluated: {manifest['num_evaluated']} | "
        f"Errors: {manifest['num_errors']}"
    )
    if manifest["num_errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
