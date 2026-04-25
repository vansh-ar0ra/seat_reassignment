"""
Run debugger for the Flight Rebooking Environment.

Persists per-step reward logs and terminal grader breakdowns to disk
for post-hoc debugging of agent runs.

Output structure per run:
    runs/<task_id>_<timestamp>_<episode_short_id>/
        steps.jsonl    -- one JSON object per environment step
        summary.json   -- terminal grader breakdown with per-component explanations
        report.md      -- human-readable debugging report
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .rewards import (
        GRADER_HARD_PENALTY,
        GRADER_W_CABIN_MATCH,
        GRADER_W_COVERAGE,
        GRADER_W_DEADLINE,
        GRADER_W_GROUP_INTEGRITY,
        GRADER_W_SSR_INTEGRITY,
        priority_weight,
    )
    from .tools import meets_deadline
except ImportError:
    from server.rewards import (
        GRADER_HARD_PENALTY,
        GRADER_W_CABIN_MATCH,
        GRADER_W_COVERAGE,
        GRADER_W_DEADLINE,
        GRADER_W_GROUP_INTEGRITY,
        GRADER_W_SSR_INTEGRITY,
        priority_weight,
    )
    from server.tools import meets_deadline

_CABIN_RANK = {"economy": 0, "premium_economy": 1, "business": 2}


class RunDebugger:
    """Captures and persists debug information for a single episode."""

    def __init__(self, runs_dir: Optional[str] = None, enabled: bool = True):
        self.enabled = enabled
        base = Path(runs_dir) if runs_dir else (
            Path(__file__).resolve().parent.parent / "runs"
        )
        self._runs_dir = base
        self._run_dir: Optional[Path] = None
        self._step_log: List[dict] = []

    def start(self, task_id: str, episode_id: str) -> Optional[str]:
        """Create a new run directory. Returns the directory path."""
        if not self.enabled:
            return None
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        short_id = episode_id[:8]
        dir_name = f"{task_id}_{timestamp}_{short_id}"
        self._run_dir = self._runs_dir / dir_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._step_log = []
        return str(self._run_dir)

    def log_step(
        self,
        step: int,
        tool_name: str,
        args: dict,
        tool_result: dict,
        reward: float,
        reward_reason: str,
        cumulative_reward: float,
        passengers_booked: int,
        passengers_total: int,
        done: bool,
    ) -> None:
        if not self.enabled or not self._run_dir:
            return
        entry = {
            "step": step,
            "tool_name": tool_name,
            "args": _safe_serialize(args),
            "result_status": tool_result.get("status", "unknown"),
            "result_detail": _extract_detail(tool_result),
            "reward": round(reward, 6),
            "reward_reason": reward_reason,
            "cumulative_reward": round(cumulative_reward, 6),
            "passengers_booked": passengers_booked,
            "passengers_total": passengers_total,
            "done": done,
        }
        self._step_log.append(entry)
        with open(self._run_dir / "steps.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_terminal(
        self,
        bookings: Dict[str, dict],
        passengers: Dict[str, dict],
        flights: Dict[str, dict],
        groups: Dict[str, List[str]],
        grader_score: float,
        breakdown: dict,
    ) -> None:
        if not self.enabled or not self._run_dir:
            return

        explanations = _explain_all(bookings, passengers, flights, groups)

        summary = {
            "grader_score": round(grader_score, 6),
            "breakdown": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in breakdown.items()
            },
            "weighted_contributions": {
                "coverage": round(GRADER_W_COVERAGE * breakdown["coverage_score"], 6),
                "cabin_match": round(GRADER_W_CABIN_MATCH * breakdown["cabin_match_score"], 6),
                "group_integrity": round(GRADER_W_GROUP_INTEGRITY * breakdown["group_integrity_score"], 6),
                "deadline": round(GRADER_W_DEADLINE * breakdown["deadline_score"], 6),
                "ssr_integrity": round(GRADER_W_SSR_INTEGRITY * breakdown["ssr_integrity_score"], 6),
                "hard_penalty": round(-GRADER_HARD_PENALTY * breakdown["hard_violations"], 6),
            },
            "explanations": explanations,
            "total_steps": len(self._step_log),
            "final_bookings": bookings,
        }

        with open(self._run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        _write_report(self._run_dir / "report.md", summary, self._step_log)


# ---------------------------------------------------------------------------
# Explanation generators
# ---------------------------------------------------------------------------

def _explain_all(
    bookings: Dict[str, dict],
    passengers: Dict[str, dict],
    flights: Dict[str, dict],
    groups: Dict[str, List[str]],
) -> dict:
    return {
        "coverage": _explain_coverage(bookings, passengers),
        "cabin_match": _explain_cabin_match(bookings, passengers),
        "group_integrity": _explain_group_integrity(bookings, passengers, groups),
        "deadline": _explain_deadline(bookings, passengers, flights),
        "ssr_integrity": _explain_ssr(bookings, passengers, flights),
    }


def _explain_coverage(
    bookings: Dict[str, dict], passengers: Dict[str, dict]
) -> dict:
    booked_ids = set(bookings.keys())
    all_ids = set(passengers.keys())
    missing = sorted(all_ids - booked_ids)
    return {
        "booked": len(booked_ids),
        "total": len(all_ids),
        "fraction": round(len(booked_ids) / len(all_ids), 4) if all_ids else 0,
        "missing_passengers": [
            {
                "passenger_id": pid,
                "name": passengers[pid]["name"],
                "priority_tier": passengers[pid]["priority_tier"],
            }
            for pid in missing
        ],
    }


def _explain_cabin_match(
    bookings: Dict[str, dict], passengers: Dict[str, dict]
) -> dict:
    details = []
    for pid, pax in passengers.items():
        entry: dict[str, Any] = {
            "passenger_id": pid,
            "name": pax["name"],
            "priority_tier": pax["priority_tier"],
            "priority_weight": priority_weight(pax["priority_tier"]),
            "original_cabin": pax["original_cabin"],
        }
        if pid in bookings:
            assigned = bookings[pid]["cabin"]
            entry["assigned_cabin"] = assigned
            entry["match"] = assigned == pax["original_cabin"]
            a_rank = _CABIN_RANK.get(assigned, 0)
            o_rank = _CABIN_RANK.get(pax["original_cabin"], 0)
            if assigned == pax["original_cabin"]:
                entry["change"] = "same"
            elif a_rank > o_rank:
                entry["change"] = "upgrade"
            else:
                entry["change"] = "downgrade"
        else:
            entry["assigned_cabin"] = None
            entry["match"] = False
            entry["change"] = "not_booked"
        details.append(entry)

    matches = [d for d in details if d["match"]]
    mismatches = [d for d in details if not d["match"] and d["assigned_cabin"] is not None]
    return {
        "matched_count": len(matches),
        "mismatched_count": len(mismatches),
        "not_booked_count": len(details) - len(matches) - len(mismatches),
        "details": details,
    }


def _explain_group_integrity(
    bookings: Dict[str, dict],
    passengers: Dict[str, dict],
    groups: Dict[str, List[str]],
) -> dict:
    if not groups:
        return {"groups": [], "note": "No groups in this scenario"}

    group_details = []
    for gid, member_ids in groups.items():
        integrity = passengers[member_ids[0]]["group_integrity"]
        members = []
        booked_flights: set[str] = set()
        booked_cabins: set[str] = set()
        all_booked = True

        for pid in member_ids:
            m: dict[str, Any] = {
                "passenger_id": pid,
                "name": passengers[pid]["name"],
            }
            if pid in bookings:
                m["flight_id"] = bookings[pid]["flight_id"]
                m["cabin"] = bookings[pid]["cabin"]
                booked_flights.add(bookings[pid]["flight_id"])
                booked_cabins.add(bookings[pid]["cabin"])
            else:
                m["flight_id"] = None
                m["cabin"] = None
                all_booked = False
            members.append(m)

        if not booked_flights:
            verdict = "none_booked"
            score = 0.0
            hard_violation = False
        elif not all_booked:
            verdict = "partially_booked"
            hard_violation = integrity == "hard"
            score = 0.0 if hard_violation else 0.04
        elif len(booked_flights) == 1 and len(booked_cabins) == 1:
            verdict = "same_flight_same_cabin"
            score = 0.7
            hard_violation = False
        elif len(booked_flights) == 1:
            verdict = "same_flight_diff_cabin"
            score = 0.5
            hard_violation = False
        else:
            verdict = "split_across_flights"
            hard_violation = integrity == "hard"
            score = 0.0 if hard_violation else 0.04

        group_details.append({
            "group_id": gid,
            "integrity": integrity,
            "size": len(member_ids),
            "members": members,
            "flights_used": sorted(booked_flights),
            "cabins_used": sorted(booked_cabins),
            "verdict": verdict,
            "component_score": score,
            "hard_violation": hard_violation,
        })

    return {"groups": group_details}


def _explain_deadline(
    bookings: Dict[str, dict],
    passengers: Dict[str, dict],
    flights: Dict[str, dict],
) -> dict:
    details = []
    for pid, pax in passengers.items():
        if pax["downstream_deadline"] is None:
            continue
        entry: dict[str, Any] = {
            "passenger_id": pid,
            "name": pax["name"],
            "priority_tier": pax["priority_tier"],
            "priority_weight": priority_weight(pax["priority_tier"]),
            "deadline": pax["downstream_deadline"],
        }
        if pid in bookings:
            fl = flights[bookings[pid]["flight_id"]]
            entry["booked_flight"] = bookings[pid]["flight_id"]
            entry["arrival_time"] = fl["arrival_time"]
            met = meets_deadline(fl["arrival_time"], pax["downstream_deadline"])
            entry["met"] = met
            if met:
                entry["verdict"] = "MET"
            else:
                entry["verdict"] = (
                    f"MISSED (arrives {fl['arrival_time']}, "
                    f"deadline {pax['downstream_deadline']})"
                )
        else:
            entry["booked_flight"] = None
            entry["arrival_time"] = None
            entry["met"] = False
            entry["verdict"] = "NOT_BOOKED"
        details.append(entry)

    return {
        "deadline_passengers": details,
        "count": len(details),
        "note": "Score defaults to 1.0 when no passengers have deadlines"
        if not details else None,
    }


def _explain_ssr(
    bookings: Dict[str, dict],
    passengers: Dict[str, dict],
    flights: Dict[str, dict],
) -> dict:
    details = []
    for pid, pax in passengers.items():
        if not pax["ssr_flags"]:
            continue
        entry: dict[str, Any] = {
            "passenger_id": pid,
            "name": pax["name"],
            "ssr_flags": pax["ssr_flags"],
        }
        if pid in bookings:
            fl = flights[bookings[pid]["flight_id"]]
            required = set(pax["ssr_flags"])
            supported = set(fl["supports_ssr"])
            missing = sorted(required - supported)
            entry["booked_flight"] = bookings[pid]["flight_id"]
            entry["flight_supports"] = fl["supports_ssr"]
            entry["violation"] = bool(missing)
            if missing:
                entry["missing_ssr"] = missing
                entry["verdict"] = f"VIOLATION — flight missing: {missing}"
            else:
                entry["verdict"] = "OK"
        else:
            entry["booked_flight"] = None
            entry["flight_supports"] = None
            entry["violation"] = False
            entry["verdict"] = "NOT_BOOKED (no violation counted)"
        details.append(entry)

    violations = [d for d in details if d["violation"]]
    return {
        "ssr_passengers": details,
        "violation_count": len(violations),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_serialize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


def _extract_detail(tool_result: dict) -> str:
    """Build a short human-readable summary from a tool result dict."""
    if not tool_result:
        return ""
    status = tool_result.get("status", "")
    if status == "error":
        return tool_result.get("message", "")
    # submit_plan results
    if "accepted_count" in tool_result and "rejected_count" in tool_result:
        preview = tool_result.get("plan_score_preview", "?")
        return (
            f"Plan: {tool_result['accepted_count']} accepted, "
            f"{tool_result['rejected_count']} rejected "
            f"(preview: {preview})"
        )
    if "passengers" in tool_result:
        return f"Listed {len(tool_result['passengers'])} passengers"
    if "flights" in tool_result:
        return f"Listed {len(tool_result['flights'])} flights"
    return tool_result.get("message", "success")


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_report(path: Path, summary: dict, steps: List[dict]) -> None:
    lines: list[str] = []
    bd = summary["breakdown"]
    wc = summary["weighted_contributions"]
    expl = summary["explanations"]

    lines.append("# Run Debug Report")
    lines.append("")
    lines.append(f"**Grader Score: {summary['grader_score']:.4f}**  ")
    lines.append(f"Total steps used: {summary['total_steps']}")
    lines.append("")

    # -- Score breakdown table --
    lines.append("## Score Breakdown")
    lines.append("")
    lines.append("| Component | Raw Score | Weight | Contribution |")
    lines.append("|-----------|-----------|--------|--------------|")
    lines.append(
        f"| Coverage | {bd['coverage_score']:.4f} | "
        f"{GRADER_W_COVERAGE} | {wc['coverage']:+.4f} |"
    )
    lines.append(
        f"| Cabin Match | {bd['cabin_match_score']:.4f} | "
        f"{GRADER_W_CABIN_MATCH} | {wc['cabin_match']:+.4f} |"
    )
    lines.append(
        f"| Group Integrity | {bd['group_integrity_score']:.4f} | "
        f"{GRADER_W_GROUP_INTEGRITY} | {wc['group_integrity']:+.4f} |"
    )
    lines.append(
        f"| Deadline | {bd['deadline_score']:.4f} | "
        f"{GRADER_W_DEADLINE} | {wc['deadline']:+.4f} |"
    )
    lines.append(
        f"| SSR Integrity | {bd['ssr_integrity_score']:.4f} | "
        f"{GRADER_W_SSR_INTEGRITY} | {wc['ssr_integrity']:+.4f} |"
    )
    lines.append(
        f"| Hard Violations | {bd['hard_violations']} | "
        f"-{GRADER_HARD_PENALTY}/each | {wc['hard_penalty']:+.4f} |"
    )
    lines.append("")

    # -- Coverage --
    cov = expl["coverage"]
    lines.append(f"## Coverage ({cov['booked']}/{cov['total']})")
    lines.append("")
    if cov["missing_passengers"]:
        lines.append("**Missing passengers:**")
        lines.append("")
        for m in cov["missing_passengers"]:
            lines.append(
                f"- {m['passenger_id']} ({m['name']}) — tier {m['priority_tier']}"
            )
        lines.append("")
    else:
        lines.append("All passengers booked.")
        lines.append("")

    # -- Cabin Match --
    cm = expl["cabin_match"]
    lines.append(
        f"## Cabin Match ({cm['matched_count']} matched, "
        f"{cm['mismatched_count']} mismatched, "
        f"{cm['not_booked_count']} not booked)"
    )
    lines.append("")
    lines.append("| Passenger | Tier | Weight | Original | Assigned | Change |")
    lines.append("|-----------|------|--------|----------|----------|--------|")
    for d in cm["details"]:
        assigned = d["assigned_cabin"] or "—"
        lines.append(
            f"| {d['passenger_id']} ({d['name']}) | {d['priority_tier']} | "
            f"{d['priority_weight']} | {d['original_cabin']} | "
            f"{assigned} | {d['change']} |"
        )
    lines.append("")

    # -- Group Integrity --
    gi = expl["group_integrity"]
    lines.append("## Group Integrity")
    lines.append("")
    if gi.get("note"):
        lines.append(gi["note"])
        lines.append("")
    for g in gi.get("groups", []):
        viol_marker = " **HARD VIOLATION**" if g["hard_violation"] else ""
        lines.append(
            f"### {g['group_id']} ({g['integrity']}, size {g['size']})"
        )
        lines.append("")
        lines.append(f"- Verdict: **{g['verdict']}**{viol_marker}")
        lines.append(f"- Component score: {g['component_score']}")
        lines.append(f"- Flights used: {g['flights_used']}")
        lines.append(f"- Cabins used: {g['cabins_used']}")
        lines.append("- Members:")
        for m in g["members"]:
            fid = m["flight_id"] or "not booked"
            cab = m["cabin"] or "—"
            lines.append(f"  - {m['passenger_id']} ({m['name']}) → {fid} / {cab}")
        lines.append("")

    # -- Deadlines --
    dl = expl["deadline"]
    lines.append(f"## Deadlines ({dl['count']} passengers with deadlines)")
    lines.append("")
    if dl.get("note"):
        lines.append(f"_{dl['note']}_")
        lines.append("")
    for d in dl.get("deadline_passengers", []):
        lines.append(
            f"- **{d['passenger_id']}** ({d['name']}, tier {d['priority_tier']}, "
            f"weight {d['priority_weight']}): deadline {d['deadline']} → "
            f"**{d['verdict']}**"
        )
    lines.append("")

    # -- SSR Integrity --
    ssr = expl["ssr_integrity"]
    lines.append(
        f"## SSR Integrity ({ssr['violation_count']} violations)"
    )
    lines.append("")
    for s in ssr.get("ssr_passengers", []):
        lines.append(
            f"- **{s['passenger_id']}** ({s['name']}): "
            f"needs {s['ssr_flags']} → **{s['verdict']}**"
        )
    lines.append("")

    # -- Step log --
    lines.append("## Step-by-Step Log")
    lines.append("")
    lines.append("| Step | Tool | Status | Reward | Cumulative | Detail |")
    lines.append("|------|------|--------|--------|------------|--------|")
    for s in steps:
        detail = s["result_detail"][:80] if s["result_detail"] else ""
        # Escape pipes in detail for markdown table
        detail = detail.replace("|", "\\|")
        lines.append(
            f"| {s['step']} | {s['tool_name']} | {s['result_status']} | "
            f"{s['reward']:+.4f} | {s['cumulative_reward']:.4f} | {detail} |"
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
