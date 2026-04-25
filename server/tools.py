"""
Tool functions for the Flight Rebooking Environment (plan-then-commit model).

4 tools:
  1. get_full_manifest    — return ALL passenger details in one call
  2. get_flight_inventory — return ALL flights with availability and SSR support
  3. submit_plan          — submit a complete rebooking plan (one shot, no revisions)
  4. finalize_plan        — lock in the current plan, trigger grading

Each function takes an EpisodeState object and named arguments.
All functions return a dict with a "status" key ("success" or "error").
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict

VALID_CABINS = {"economy", "premium_economy", "business"}


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_time(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def meets_deadline(arrival_time: str, deadline: str) -> bool:
    """True if flight arrives at or before the deadline."""
    return parse_time(arrival_time) <= parse_time(deadline)


# ---------------------------------------------------------------------------
# Tool 1: get_full_manifest
# ---------------------------------------------------------------------------

def tool_get_full_manifest(ep) -> dict:
    """Return full details for ALL passengers in one call."""
    passengers_list = []
    for pid, pax in ep.passengers.items():
        entry = {
            "passenger_id": pid,
            "name": pax["name"],
            "priority_tier": pax["priority_tier"],
            "original_cabin": pax["original_cabin"],
            "group_id": pax["group_id"],
            "group_integrity": pax["group_integrity"],
            "group_size": pax["group_size"],
            "ssr_flags": pax["ssr_flags"],
            "downstream_deadline": pax["downstream_deadline"],
        }
        if pid in ep.bookings:
            entry["current_booking"] = {
                "flight_id": ep.bookings[pid]["flight_id"],
                "cabin": ep.bookings[pid]["cabin"],
            }
        passengers_list.append(entry)

    return {"status": "success", "passengers": passengers_list}


# ---------------------------------------------------------------------------
# Tool 2: get_flight_inventory
# ---------------------------------------------------------------------------

def tool_get_flight_inventory(ep) -> dict:
    """Return all flights with current availability and SSR support."""
    flights_list = []
    for fid, fl in ep.flights.items():
        flights_list.append({
            "flight_id": fid,
            "departure_time": fl["departure_time"],
            "arrival_time": fl["arrival_time"],
            "cabin_availability": dict(ep.flight_availability[fid]),
            "supports_ssr": fl["supports_ssr"],
        })

    return {"status": "success", "flights": flights_list}


# ---------------------------------------------------------------------------
# Tool 3: submit_plan
# ---------------------------------------------------------------------------

def tool_submit_plan(ep, assignments: "Dict[str, dict]", reward_computer=None) -> dict:
    """
    Submit a complete rebooking plan. Validated atomically.
    Only ONE submission allowed per episode -- no revisions.

    assignments: dict mapping passenger_id -> {"flight_id": str, "cabin": str}

    Returns per-passenger results, group integrity results, and a grader preview.
    """
    # 0. Check if plan already submitted
    if ep.plan_submitted:
        return {
            "status": "error",
            "message": "Plan already submitted. No revisions allowed.",
        }

    # 1. Reset bookings and availability to initial state
    ep.bookings = {}
    ep.flight_availability = copy.deepcopy(ep.initial_availability)

    per_passenger = []
    constraint_violations = []

    # 2. Process each assignment
    for passenger_id, assignment in assignments.items():
        # Skip passengers with null/None assignment (agent chose not to book them)
        if assignment is None:
            per_passenger.append({
                "passenger_id": passenger_id,
                "flight_id": None,
                "cabin": None,
                "status": "skipped",
                "reason": "No assignment provided",
            })
            continue
        flight_id = assignment.get("flight_id", "")
        cabin = assignment.get("cabin", "")
        result_entry = {
            "passenger_id": passenger_id,
            "flight_id": flight_id,
            "cabin": cabin,
        }

        # 2a. Validate passenger exists
        if passenger_id not in ep.passengers:
            result_entry["status"] = "rejected"
            result_entry["reason"] = f"Passenger {passenger_id} does not exist"
            per_passenger.append(result_entry)
            continue

        # 2b. Validate flight exists
        if flight_id not in ep.flights:
            result_entry["status"] = "rejected"
            result_entry["reason"] = f"Flight {flight_id} does not exist"
            per_passenger.append(result_entry)
            continue

        # 2c. Validate cabin
        if cabin not in VALID_CABINS:
            result_entry["status"] = "rejected"
            result_entry["reason"] = f"Invalid cabin '{cabin}'. Must be one of: {sorted(VALID_CABINS)}"
            per_passenger.append(result_entry)
            continue

        # 2d. Check cabin availability
        avail = ep.flight_availability[flight_id].get(cabin, 0)
        if avail <= 0:
            result_entry["status"] = "rejected"
            result_entry["reason"] = f"No {cabin} seats available on {flight_id}"
            per_passenger.append(result_entry)
            continue

        pax = ep.passengers[passenger_id]
        fl = ep.flights[flight_id]

        # 2e. SSR compatibility
        if pax["ssr_flags"]:
            supported = set(fl["supports_ssr"])
            required = set(pax["ssr_flags"])
            missing = required - supported
            if missing:
                result_entry["status"] = "rejected"
                result_entry["reason"] = (
                    f"Flight {flight_id} does not support SSR: {sorted(missing)}. "
                    f"Passenger requires {sorted(required)}, "
                    f"flight supports {sorted(supported)}"
                )
                per_passenger.append(result_entry)
                continue

        # 2f. Deadline check
        if pax["downstream_deadline"]:
            if not meets_deadline(fl["arrival_time"], pax["downstream_deadline"]):
                result_entry["status"] = "rejected"
                result_entry["reason"] = (
                    f"Flight {flight_id} arrives at {fl['arrival_time']} "
                    f"which is after deadline {pax['downstream_deadline']}"
                )
                per_passenger.append(result_entry)
                continue

        # 2g. All validations passed — accept
        ep.flight_availability[flight_id][cabin] -= 1
        ep.bookings[passenger_id] = {"flight_id": flight_id, "cabin": cabin}
        result_entry["status"] = "accepted"
        result_entry["reason"] = "OK"
        per_passenger.append(result_entry)

    # 3. Post-validation group integrity check
    group_results = []
    for gid, member_ids in ep.groups.items():
        integrity = ep.passengers[member_ids[0]]["group_integrity"]
        members_flights = {}
        for mid in member_ids:
            if mid in ep.bookings:
                members_flights[mid] = ep.bookings[mid]["flight_id"]
            else:
                members_flights[mid] = None

        booked_flights = set(f for f in members_flights.values() if f is not None)
        all_booked = all(f is not None for f in members_flights.values())

        if not booked_flights:
            verdict = "none_booked"
        elif not all_booked:
            verdict = "partially_booked"
            if integrity == "hard":
                constraint_violations.append(
                    f"Hard group {gid}: not all members booked"
                )
        elif len(booked_flights) == 1:
            verdict = "together"
        else:
            verdict = "split_across_flights"
            if integrity == "hard":
                constraint_violations.append(
                    f"Hard group {gid}: members split across flights {sorted(booked_flights)}"
                )

        group_results.append({
            "group_id": gid,
            "integrity": integrity,
            "verdict": verdict,
            "members_flights": members_flights,
        })

    # 4. Compute grader score preview
    accepted_count = sum(1 for p in per_passenger if p["status"] == "accepted")
    rejected_count = sum(1 for p in per_passenger if p["status"] == "rejected")

    preview = 0.0
    if reward_computer is not None:
        preview = reward_computer.grader_score(
            ep.bookings, ep.passengers, ep.flights, ep.groups
        )

    # 5. Store preview and mark plan as submitted
    ep.last_plan_preview = preview
    ep.plan_submitted = True

    return {
        "status": "success",
        "per_passenger": per_passenger,
        "group_results": group_results,
        "plan_score_preview": round(preview, 6),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total": len(assignments),
        "constraint_violations": constraint_violations,
    }


# ---------------------------------------------------------------------------
# Tool 4: finalize_plan
# ---------------------------------------------------------------------------

def tool_finalize_plan(ep) -> dict:
    """
    End the episode and trigger grading.
    Returns a warning if no plan has been submitted.
    """
    if not ep.plan_submitted:
        ep.done = True
        return {
            "status": "success",
            "message": "Finalized without a submitted plan. Score will be low.",
        }
    ep.done = True
    return {"status": "success", "message": "Plan finalized. Grading in progress."}
