"""
Tool functions for the Airline Seat Reassignment Environment.

Each function takes a `state` object and named arguments. The state object is
expected to have the following attributes (populated at reset() time):

    state.assignments       pd.DataFrame indexed on passenger_id;
                            columns: seat_ac1, seat_ac2 (NaN until assigned)
    state.passengers_by_id  Dict[str, dict]  passenger_id → passenger row
    state.ac1_seat_set      Set[str]         valid AC-1 seat IDs
    state.ac2_seat_set      Set[str]         valid AC-2 seat IDs
    state.ac1_seat_info     Dict[str, dict]  AC-1 seat_id → {cabin, seat_type[, extra_legroom]}
    state.ac2_seat_info     Dict[str, dict]  AC-2 seat_id → {cabin, seat_type[, extra_legroom]}
    state.fetched_seats     Set[str]         AC-1 seats already queried

All functions return a dict with a "status" key ("success" or "error").
On error, a "message" key describes what went wrong.
No exceptions are raised for invalid inputs; they are returned as error dicts.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _preference_satisfied(passenger: dict, seat_info: dict) -> tuple:
    """
    Returns (window_satisfied, legroom_satisfied).

    Each element is:
      True  — passenger paid for this pref and it is satisfied
      False — passenger paid for this pref and it is NOT satisfied
      None  — passenger did not pay for this pref (not applicable)

    Gracefully handles passengers without paid_legroom (easy/medium tasks):
      passenger.get("paid_legroom", False) defaults to False → legroom=None.
    """
    window_paid  = passenger.get("paid_window",  False)
    legroom_paid = passenger.get("paid_legroom", False)

    window  = (seat_info.get("seat_type", "") == "window") if window_paid  else None
    legroom = bool(seat_info.get("extra_legroom", False))  if legroom_paid else None

    return window, legroom


# ---------------------------------------------------------------------------
# Tool 1: get_passenger_details
# ---------------------------------------------------------------------------

def tool_get_passenger_details(state, seat_id: str) -> dict:
    """
    Return details for the passenger currently (or previously) in an AC-1 seat.

    Side effect: adds seat_id to state.fetched_seats when the passenger is
    still on AC-1.
    """
    if seat_id not in state.ac1_seat_set:
        return {
            "status":  "error",
            "message": f"Seat {seat_id} does not exist on AC-1",
        }

    # Find the passenger whose AC-1 seat matches (boolean index on non-key column)
    mask = state.assignments["seat_ac1"] == seat_id
    row = state.assignments.loc[mask]
    passenger_id = row.index[0]
    seat_ac2 = row.loc[passenger_id, "seat_ac2"]

    if pd.notna(seat_ac2):
        return {
            "status":  "error",
            "message": (
                f"Seat {seat_id} on AC-1 is now empty — "
                f"passenger already reassigned to AC-2 seat {seat_ac2}"
            ),
        }

    pax = state.passengers_by_id[passenger_id]
    state.fetched_seats.add(seat_id)

    result = {
        "status":           "success",
        "passenger_id":     passenger_id,
        "name":             pax["name"],
        "cabin":            pax["cabin"],
        "paid_window":      pax.get("paid_window",  False),
        "paid_legroom":     pax.get("paid_legroom", False),
        "current_seat_ac1": seat_id,
        "current_seat_type": state.ac1_seat_info[seat_id]["seat_type"],
    }
    # Include extra_legroom of current AC-1 seat when available
    if "extra_legroom" in state.ac1_seat_info[seat_id]:
        result["current_seat_extra_legroom"] = state.ac1_seat_info[seat_id]["extra_legroom"]

    return result


# ---------------------------------------------------------------------------
# Tool 2: assign_seat
# ---------------------------------------------------------------------------

def tool_assign_seat(state, passenger_id: str, target_seat_id: str) -> dict:
    """
    Assign a passenger from their AC-1 seat to a specific AC-2 seat.

    Mutates: state.assignments (sets seat_ac2 for the passenger's row).
    """
    # 1. Passenger must exist
    if passenger_id not in state.passengers_by_id:
        return {
            "status":  "error",
            "message": f"Passenger {passenger_id} does not exist",
        }

    # 2. Passenger must not already be on AC-2
    seat_ac2 = state.assignments.loc[passenger_id, "seat_ac2"]
    if pd.notna(seat_ac2):
        return {
            "status":  "error",
            "message": f"Passenger {passenger_id} is already assigned to AC-2 seat {seat_ac2}",
        }

    # 3. Target seat must exist on AC-2
    if target_seat_id not in state.ac2_seat_set:
        return {
            "status":  "error",
            "message": f"Seat {target_seat_id} does not exist on AC-2",
        }

    # 4. Target seat must be unoccupied
    occupied_mask = state.assignments["seat_ac2"] == target_seat_id
    occupied = state.assignments.loc[occupied_mask]
    if not occupied.empty:
        occupant_id = occupied.index[0]
        return {
            "status":  "error",
            "message": f"Seat {target_seat_id} on AC-2 is already occupied by {occupant_id}",
        }

    pax      = state.passengers_by_id[passenger_id]
    from_seat = state.assignments.loc[passenger_id, "seat_ac1"]
    ac2_seat  = state.ac2_seat_info[target_seat_id]

    cabin_match = pax["cabin"] == ac2_seat["cabin"]
    window_pref, legroom_pref = _preference_satisfied(pax, ac2_seat)

    # Commit the assignment.
    # seat_ac2 is loaded as float64 (all-NaN column from CSV); upcast to object
    # so it can hold string seat IDs alongside NaN values.
    if state.assignments["seat_ac2"].dtype != object:
        state.assignments["seat_ac2"] = state.assignments["seat_ac2"].astype(object)
    state.assignments.loc[passenger_id, "seat_ac2"] = target_seat_id

    return {
        "status":                       "success",
        "passenger_id":                 passenger_id,
        "from_seat_ac1":                from_seat,
        "to_seat_ac2":                  target_seat_id,
        "cabin_match":                  cabin_match,
        "window_preference_satisfied":  window_pref,
        "legroom_preference_satisfied": legroom_pref,
    }


# ---------------------------------------------------------------------------
# Tool 3: swap_seats
# ---------------------------------------------------------------------------

def tool_swap_seats(state, passenger_id_1: str, passenger_id_2: str) -> dict:
    """
    Swap the AC-2 seat assignments of two already-assigned passengers.

    Mutates: state.assignments (exchanges seat_ac2 values for both rows).
    """
    # 1. Both passengers must exist
    for pid in (passenger_id_1, passenger_id_2):
        if pid not in state.passengers_by_id:
            return {
                "status":  "error",
                "message": f"Passenger {pid} does not exist",
            }

    # 2. Must be two distinct passengers
    if passenger_id_1 == passenger_id_2:
        return {
            "status":  "error",
            "message": "Cannot swap a passenger with themselves",
        }

    # 3. Both must already be assigned to AC-2
    for pid in (passenger_id_1, passenger_id_2):
        if pd.isna(state.assignments.loc[pid, "seat_ac2"]):
            return {
                "status":  "error",
                "message": f"Passenger {pid} is not yet assigned to AC-2",
            }

    seat1 = state.assignments.loc[passenger_id_1, "seat_ac2"]
    seat2 = state.assignments.loc[passenger_id_2, "seat_ac2"]

    pax1 = state.passengers_by_id[passenger_id_1]
    pax2 = state.passengers_by_id[passenger_id_2]

    # Metrics before the swap
    cabin_match_1_before = pax1["cabin"] == state.ac2_seat_info[seat1]["cabin"]
    cabin_match_2_before = pax2["cabin"] == state.ac2_seat_info[seat2]["cabin"]
    win1_before,  leg1_before  = _preference_satisfied(pax1, state.ac2_seat_info[seat1])
    win2_before,  leg2_before  = _preference_satisfied(pax2, state.ac2_seat_info[seat2])

    # Execute the swap
    state.assignments.loc[passenger_id_1, "seat_ac2"] = seat2
    state.assignments.loc[passenger_id_2, "seat_ac2"] = seat1

    # Metrics after the swap (pax1 is now in seat2, pax2 in seat1)
    cabin_match_1_after = pax1["cabin"] == state.ac2_seat_info[seat2]["cabin"]
    cabin_match_2_after = pax2["cabin"] == state.ac2_seat_info[seat1]["cabin"]
    win1_after,  leg1_after  = _preference_satisfied(pax1, state.ac2_seat_info[seat2])
    win2_after,  leg2_after  = _preference_satisfied(pax2, state.ac2_seat_info[seat1])

    return {
        "status": "success",
        "swap": [
            {"passenger_id": passenger_id_1, "from_seat": seat1, "to_seat": seat2},
            {"passenger_id": passenger_id_2, "from_seat": seat2, "to_seat": seat1},
        ],
        "improvement": {
            passenger_id_1: {
                "cabin_match_before":            cabin_match_1_before,
                "cabin_match_after":             cabin_match_1_after,
                "window_preference_before":      win1_before,
                "window_preference_after":       win1_after,
                "legroom_preference_before":     leg1_before,
                "legroom_preference_after":      leg1_after,
            },
            passenger_id_2: {
                "cabin_match_before":            cabin_match_2_before,
                "cabin_match_after":             cabin_match_2_after,
                "window_preference_before":      win2_before,
                "window_preference_after":       win2_after,
                "legroom_preference_before":     leg2_before,
                "legroom_preference_after":      leg2_after,
            },
        },
    }
