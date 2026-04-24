"""
Tool functions for the Flight Rebooking Environment.

Each function takes an EpisodeState object and named arguments.
The EpisodeState is expected to have:

    ep.passengers              Dict[str, dict]  passenger_id -> full record
    ep.flights                 Dict[str, dict]  flight_id -> full record
    ep.groups                  Dict[str, List[str]]  group_id -> [passenger_ids]
    ep.bookings                Dict[str, dict]  passenger_id -> {flight_id, cabin}
    ep.flight_availability     Dict[str, Dict[str, int]]  flight_id -> {cabin: count}
    ep.passenger_details_fetched  Set[str]  passenger_ids whose details have been fetched
    ep.info_calls              Dict[str, int]  tool_name -> call count
    ep.last_booking_step       int
    ep.step_count              int
    ep.done                    bool

All functions return a dict with a "status" key ("success" or "error").
On error, a "message" key describes what went wrong.
No exceptions are raised for invalid inputs; they are returned as error dicts.
"""

from __future__ import annotations

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
# Tool 1: list_passengers
# ---------------------------------------------------------------------------

def tool_list_passengers(ep) -> dict:
    """Return a lightweight summary of all passengers."""
    ep.info_calls["list_passengers"] = ep.info_calls.get("list_passengers", 0) + 1

    summary = []
    for pid, pax in ep.passengers.items():
        summary.append({
            "passenger_id": pid,
            "priority_tier": pax["priority_tier"],
            "group_id": pax["group_id"],
            "has_ssr": len(pax["ssr_flags"]) > 0,
            "has_deadline": pax["downstream_deadline"] is not None,
            "booked": pid in ep.bookings,
        })

    return {"status": "success", "passengers": summary}


# ---------------------------------------------------------------------------
# Tool 2: get_passenger_details
# ---------------------------------------------------------------------------

def tool_get_passenger_details(ep, passenger_id: str) -> dict:
    """Return the full record for a single passenger."""
    if passenger_id not in ep.passengers:
        return {
            "status": "error",
            "message": f"Passenger {passenger_id} does not exist",
        }

    ep.passenger_details_fetched.add(passenger_id)

    pax = ep.passengers[passenger_id]
    result = {
        "status": "success",
        "passenger_id": passenger_id,
        "name": pax["name"],
        "priority_tier": pax["priority_tier"],
        "original_cabin": pax["original_cabin"],
        "group_id": pax["group_id"],
        "group_integrity": pax["group_integrity"],
        "group_size": pax["group_size"],
        "ssr_flags": pax["ssr_flags"],
        "downstream_deadline": pax["downstream_deadline"],
    }

    if passenger_id in ep.bookings:
        booking = ep.bookings[passenger_id]
        result["current_booking"] = {
            "flight_id": booking["flight_id"],
            "cabin": booking["cabin"],
        }

    return result


# ---------------------------------------------------------------------------
# Tool 3: list_alternative_flights
# ---------------------------------------------------------------------------

def tool_list_alternative_flights(ep) -> dict:
    """Return all alternative flights with current availability."""
    ep.info_calls["list_alternative_flights"] = (
        ep.info_calls.get("list_alternative_flights", 0) + 1
    )

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
# Tool 4: get_flight_details
# ---------------------------------------------------------------------------

def tool_get_flight_details(ep, flight_id: str) -> dict:
    """Return full details for a single flight including current availability."""
    ep.info_calls["get_flight_details"] = (
        ep.info_calls.get("get_flight_details", 0) + 1
    )

    if flight_id not in ep.flights:
        return {
            "status": "error",
            "message": f"Flight {flight_id} does not exist",
        }

    fl = ep.flights[flight_id]
    return {
        "status": "success",
        "flight_id": flight_id,
        "departure_time": fl["departure_time"],
        "arrival_time": fl["arrival_time"],
        "cabin_availability": dict(ep.flight_availability[flight_id]),
        "supports_ssr": fl["supports_ssr"],
    }


# ---------------------------------------------------------------------------
# Tool 5: book_passenger
# ---------------------------------------------------------------------------

def tool_book_passenger(ep, passenger_id: str, flight_id: str, cabin: str) -> dict:
    """
    Book a single passenger onto a flight in the specified cabin.

    Validation chain:
    1. Passenger exists
    2. Passenger not already booked
    3. Flight exists
    4. Cabin is valid
    5. Cabin has availability > 0
    6. Flight supports all of passenger's SSR flags
    7. If downstream_deadline, arrival_time <= deadline
    8. If hard group member, warn (should use book_group)

    On success: decrements availability, adds to bookings.
    """
    # 1. Passenger exists
    if passenger_id not in ep.passengers:
        return {
            "status": "error",
            "message": f"Passenger {passenger_id} does not exist",
        }

    # 2. Not already booked
    if passenger_id in ep.bookings:
        existing = ep.bookings[passenger_id]
        return {
            "status": "error",
            "message": (
                f"Passenger {passenger_id} is already booked on "
                f"{existing['flight_id']} in {existing['cabin']}"
            ),
        }

    # 3. Flight exists
    if flight_id not in ep.flights:
        return {
            "status": "error",
            "message": f"Flight {flight_id} does not exist",
        }

    # 4. Valid cabin
    if cabin not in VALID_CABINS:
        return {
            "status": "error",
            "message": f"Invalid cabin '{cabin}'. Must be one of: {sorted(VALID_CABINS)}",
        }

    # 5. Availability
    avail = ep.flight_availability[flight_id].get(cabin, 0)
    if avail <= 0:
        return {
            "status": "error",
            "message": f"No {cabin} seats available on {flight_id}",
        }

    pax = ep.passengers[passenger_id]
    fl = ep.flights[flight_id]

    # 6. SSR compatibility
    if pax["ssr_flags"]:
        supported = set(fl["supports_ssr"])
        required = set(pax["ssr_flags"])
        missing = required - supported
        if missing:
            return {
                "status": "error",
                "message": (
                    f"Flight {flight_id} does not support SSR: {sorted(missing)}. "
                    f"Passenger {passenger_id} requires {sorted(required)}, "
                    f"flight supports {sorted(supported)}"
                ),
            }

    # 7. Deadline check
    deadline_met = None
    if pax["downstream_deadline"]:
        if not meets_deadline(fl["arrival_time"], pax["downstream_deadline"]):
            return {
                "status": "error",
                "message": (
                    f"Flight {flight_id} arrives at {fl['arrival_time']} "
                    f"which is after passenger {passenger_id}'s deadline "
                    f"of {pax['downstream_deadline']}"
                ),
            }
        deadline_met = True

    # 8. Hard group warning
    warnings = []
    if pax["group_id"] and pax["group_integrity"] == "hard":
        warnings.append(
            f"Passenger {passenger_id} is in hard group {pax['group_id']}. "
            f"Consider using book_group to keep the group together."
        )

    # --- Commit booking ---
    ep.flight_availability[flight_id][cabin] -= 1
    ep.bookings[passenger_id] = {"flight_id": flight_id, "cabin": cabin}
    ep.last_booking_step = ep.step_count

    cabin_match = pax["original_cabin"] == cabin

    result = {
        "status": "success",
        "passenger_id": passenger_id,
        "flight_id": flight_id,
        "cabin": cabin,
        "cabin_match": cabin_match,
        "original_cabin": pax["original_cabin"],
    }
    if deadline_met is not None:
        result["deadline_met"] = deadline_met
    if warnings:
        result["warnings"] = warnings

    return result


# ---------------------------------------------------------------------------
# Tool 6: book_group
# ---------------------------------------------------------------------------

def tool_book_group(
    ep,
    group_id: str,
    flight_id: str,
    cabin_assignments: "Dict[str, str]",
) -> dict:
    """
    Book an entire group onto a single flight. Atomic — all or none.

    cabin_assignments maps passenger_id -> cabin for each group member.

    Validation chain:
    1. Group exists
    2. All group members present in cabin_assignments and none already booked
    3. Flight exists
    4. All cabins valid
    5. Sufficient capacity for all members
    6. Flight supports SSR flags of all group members
    7. Deadline check for all members
    """
    # 1. Group exists
    if group_id not in ep.groups:
        return {
            "status": "error",
            "message": f"Group {group_id} does not exist",
        }

    member_ids = ep.groups[group_id]

    # 2. All members present and none already booked
    provided_ids = set(cabin_assignments.keys())
    expected_ids = set(member_ids)
    if provided_ids != expected_ids:
        missing = expected_ids - provided_ids
        extra = provided_ids - expected_ids
        parts = []
        if missing:
            parts.append(f"missing assignments for: {sorted(missing)}")
        if extra:
            parts.append(f"unknown members: {sorted(extra)}")
        return {
            "status": "error",
            "message": f"cabin_assignments mismatch for group {group_id}: {'; '.join(parts)}",
        }

    for pid in member_ids:
        if pid in ep.bookings:
            existing = ep.bookings[pid]
            return {
                "status": "error",
                "message": (
                    f"Group member {pid} is already booked on "
                    f"{existing['flight_id']} in {existing['cabin']}"
                ),
            }

    # 3. Flight exists
    if flight_id not in ep.flights:
        return {
            "status": "error",
            "message": f"Flight {flight_id} does not exist",
        }

    fl = ep.flights[flight_id]

    # 4. All cabins valid
    for pid, cabin in cabin_assignments.items():
        if cabin not in VALID_CABINS:
            return {
                "status": "error",
                "message": (
                    f"Invalid cabin '{cabin}' for passenger {pid}. "
                    f"Must be one of: {sorted(VALID_CABINS)}"
                ),
            }

    # 5. Capacity check — compute total demand per cabin
    cabin_demand: dict[str, int] = {}
    for pid, cabin in cabin_assignments.items():
        cabin_demand[cabin] = cabin_demand.get(cabin, 0) + 1

    for cabin, needed in cabin_demand.items():
        available = ep.flight_availability[flight_id].get(cabin, 0)
        if needed > available:
            return {
                "status": "error",
                "message": (
                    f"Not enough {cabin} seats on {flight_id}: "
                    f"need {needed}, available {available}"
                ),
            }

    # 6. SSR check for all members
    supported = set(fl["supports_ssr"])
    for pid in member_ids:
        pax = ep.passengers[pid]
        if pax["ssr_flags"]:
            required = set(pax["ssr_flags"])
            missing = required - supported
            if missing:
                return {
                    "status": "error",
                    "message": (
                        f"Flight {flight_id} does not support SSR: {sorted(missing)}. "
                        f"Group member {pid} requires {sorted(required)}, "
                        f"flight supports {sorted(supported)}"
                    ),
                }

    # 7. Deadline check for all members
    for pid in member_ids:
        pax = ep.passengers[pid]
        if pax["downstream_deadline"]:
            if not meets_deadline(fl["arrival_time"], pax["downstream_deadline"]):
                return {
                    "status": "error",
                    "message": (
                        f"Flight {flight_id} arrives at {fl['arrival_time']} "
                        f"which is after group member {pid}'s deadline "
                        f"of {pax['downstream_deadline']}"
                    ),
                }

    # --- Commit all bookings atomically ---
    booked = []
    for pid, cabin in cabin_assignments.items():
        ep.flight_availability[flight_id][cabin] -= 1
        ep.bookings[pid] = {"flight_id": flight_id, "cabin": cabin}
        pax = ep.passengers[pid]
        booked.append({
            "passenger_id": pid,
            "cabin": cabin,
            "cabin_match": pax["original_cabin"] == cabin,
            "original_cabin": pax["original_cabin"],
        })

    ep.last_booking_step = ep.step_count

    return {
        "status": "success",
        "group_id": group_id,
        "flight_id": flight_id,
        "booked": booked,
    }


# ---------------------------------------------------------------------------
# Tool 7: finalize_plan
# ---------------------------------------------------------------------------

def tool_finalize_plan(ep) -> dict:
    """
    End the episode and trigger grading.

    Returns a status dict. The environment.step() method handles
    computing and attaching the grader score.
    """
    ep.done = True
    return {"status": "success", "message": "Plan finalized. Grading in progress."}
