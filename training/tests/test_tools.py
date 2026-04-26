"""CPU-only unit tests for server/tools.py — tool dispatch and validation logic.

Covers:
  - parse_time: HH:MM → minutes conversion
  - meets_deadline: arrival vs deadline comparison
  - tool_get_full_manifest: correct passenger data returned
  - tool_get_flight_inventory: correct flight data returned
  - tool_submit_plan: acceptance, rejection, capacity decrement, SSR checks,
    deadline checks, group integrity detection, duplicate submit blocking
  - tool_finalize_plan: done flag, with/without plan
  - Edge cases: unknown passenger, unknown flight, invalid cabin, null assignment

No GPU or torch needed.
"""

import copy
import pytest
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from server.tools import (
    parse_time,
    meets_deadline,
    tool_get_full_manifest,
    tool_get_flight_inventory,
    tool_submit_plan,
    tool_finalize_plan,
    VALID_CABINS,
)


# ───────────────────────────────────────────────────────────
# Minimal EpisodeState stub (matches the interface tools.py uses)
# ───────────────────────────────────────────────────────────

@dataclass
class _StubEpisode:
    passengers: Dict[str, dict] = field(default_factory=dict)
    flights: Dict[str, dict] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    bookings: Dict[str, dict] = field(default_factory=dict)
    flight_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)
    initial_availability: Dict[str, Dict[str, int]] = field(default_factory=dict)
    plan_submitted: bool = False
    last_plan_preview: float = 0.0
    done: bool = False


def _make_episode() -> _StubEpisode:
    """Build a small test episode with 3 passengers, 2 flights."""
    passengers = {
        "PAX-1": {
            "passenger_id": "PAX-1",
            "name": "Alice",
            "priority_tier": 1,
            "original_cabin": "business",
            "group_id": None,
            "group_integrity": None,
            "group_size": None,
            "ssr_flags": [],
            "downstream_deadline": None,
        },
        "PAX-2": {
            "passenger_id": "PAX-2",
            "name": "Bob",
            "priority_tier": 3,
            "original_cabin": "economy",
            "group_id": "GRP-1",
            "group_integrity": "hard",
            "group_size": 2,
            "ssr_flags": ["WCHR"],
            "downstream_deadline": "14:00",
        },
        "PAX-3": {
            "passenger_id": "PAX-3",
            "name": "Carol",
            "priority_tier": 3,
            "original_cabin": "economy",
            "group_id": "GRP-1",
            "group_integrity": "hard",
            "group_size": 2,
            "ssr_flags": [],
            "downstream_deadline": None,
        },
    }
    flights = {
        "FL-201": {
            "flight_id": "FL-201",
            "departure_time": "09:00",
            "arrival_time": "12:00",
            "supports_ssr": ["WCHR", "UM"],
            "cabin_availability": {"economy": 5, "business": 2},
        },
        "FL-202": {
            "flight_id": "FL-202",
            "departure_time": "14:00",
            "arrival_time": "17:00",
            "supports_ssr": [],
            "cabin_availability": {"economy": 3, "business": 1},
        },
    }
    groups = {"GRP-1": ["PAX-2", "PAX-3"]}
    avail = {
        "FL-201": {"economy": 5, "business": 2},
        "FL-202": {"economy": 3, "business": 1},
    }
    return _StubEpisode(
        passengers=passengers,
        flights=flights,
        groups=groups,
        bookings={},
        flight_availability=copy.deepcopy(avail),
        initial_availability=copy.deepcopy(avail),
    )


# ───────────────────────────────────────────────────────────
# parse_time
# ───────────────────────────────────────────────────────────

class TestParseTime:
    def test_midnight(self):
        assert parse_time("00:00") == 0

    def test_noon(self):
        assert parse_time("12:00") == 720

    def test_arbitrary(self):
        assert parse_time("14:30") == 870

    def test_end_of_day(self):
        assert parse_time("23:59") == 1439


# ───────────────────────────────────────────────────────────
# meets_deadline
# ───────────────────────────────────────────────────────────

class TestMeetsDeadline:
    def test_arrival_before_deadline(self):
        assert meets_deadline("12:00", "14:00") is True

    def test_arrival_at_deadline(self):
        assert meets_deadline("14:00", "14:00") is True

    def test_arrival_after_deadline(self):
        assert meets_deadline("15:00", "14:00") is False

    def test_one_minute_late(self):
        assert meets_deadline("14:01", "14:00") is False


# ───────────────────────────────────────────────────────────
# tool_get_full_manifest
# ───────────────────────────────────────────────────────────

class TestGetFullManifest:
    def test_returns_all_passengers(self):
        ep = _make_episode()
        result = tool_get_full_manifest(ep)
        assert result["status"] == "success"
        assert len(result["passengers"]) == 3

    def test_passenger_fields_present(self):
        ep = _make_episode()
        result = tool_get_full_manifest(ep)
        entry = result["passengers"][0]
        for field in [
            "passenger_id", "name", "priority_tier", "original_cabin",
            "group_id", "group_integrity", "group_size", "ssr_flags",
            "downstream_deadline",
        ]:
            assert field in entry, f"Missing field: {field}"

    def test_no_current_booking_initially(self):
        ep = _make_episode()
        result = tool_get_full_manifest(ep)
        for entry in result["passengers"]:
            assert "current_booking" not in entry

    def test_shows_booking_if_present(self):
        ep = _make_episode()
        ep.bookings["PAX-1"] = {"flight_id": "FL-201", "cabin": "business"}
        result = tool_get_full_manifest(ep)
        pax1 = [p for p in result["passengers"] if p["passenger_id"] == "PAX-1"][0]
        assert "current_booking" in pax1
        assert pax1["current_booking"]["flight_id"] == "FL-201"


# ───────────────────────────────────────────────────────────
# tool_get_flight_inventory
# ───────────────────────────────────────────────────────────

class TestGetFlightInventory:
    def test_returns_all_flights(self):
        ep = _make_episode()
        result = tool_get_flight_inventory(ep)
        assert result["status"] == "success"
        assert len(result["flights"]) == 2

    def test_flight_fields_present(self):
        ep = _make_episode()
        result = tool_get_flight_inventory(ep)
        entry = result["flights"][0]
        for field in ["flight_id", "departure_time", "arrival_time", "cabin_availability", "supports_ssr"]:
            assert field in entry, f"Missing field: {field}"


# ───────────────────────────────────────────────────────────
# tool_submit_plan
# ───────────────────────────────────────────────────────────

class TestSubmitPlan:
    def test_valid_plan_accepted(self):
        ep = _make_episode()
        assignments = {
            "PAX-1": {"flight_id": "FL-201", "cabin": "business"},
            "PAX-2": {"flight_id": "FL-201", "cabin": "economy"},
            "PAX-3": {"flight_id": "FL-201", "cabin": "economy"},
        }
        result = tool_submit_plan(ep, assignments)
        assert result["status"] == "success"
        assert result["accepted_count"] == 3
        assert result["rejected_count"] == 0

    def test_unknown_passenger_rejected(self):
        ep = _make_episode()
        assignments = {"PAX-FAKE": {"flight_id": "FL-201", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        assert result["rejected_count"] == 1
        rejected = [p for p in result["per_passenger"] if p["status"] == "rejected"]
        assert "does not exist" in rejected[0]["reason"]

    def test_unknown_flight_rejected(self):
        ep = _make_episode()
        assignments = {"PAX-1": {"flight_id": "FL-999", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        assert result["rejected_count"] == 1

    def test_invalid_cabin_rejected(self):
        ep = _make_episode()
        assignments = {"PAX-1": {"flight_id": "FL-201", "cabin": "first_class"}}
        result = tool_submit_plan(ep, assignments)
        rejected = [p for p in result["per_passenger"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert "Invalid cabin" in rejected[0]["reason"]

    def test_capacity_overflow_rejected(self):
        ep = _make_episode()
        ep.flight_availability["FL-201"]["business"] = 1
        ep.initial_availability["FL-201"]["business"] = 1
        assignments = {
            "PAX-1": {"flight_id": "FL-201", "cabin": "business"},
            "PAX-3": {"flight_id": "FL-201", "cabin": "business"},
        }
        result = tool_submit_plan(ep, assignments)
        assert result["accepted_count"] == 1
        assert result["rejected_count"] == 1

    def test_ssr_incompatible_rejected(self):
        """PAX-2 has WCHR. FL-202 does NOT support WCHR."""
        ep = _make_episode()
        assignments = {"PAX-2": {"flight_id": "FL-202", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        rejected = [p for p in result["per_passenger"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert "SSR" in rejected[0]["reason"] or "ssr" in rejected[0]["reason"].lower()

    def test_ssr_compatible_accepted(self):
        """PAX-2 has WCHR. FL-201 supports WCHR."""
        ep = _make_episode()
        assignments = {"PAX-2": {"flight_id": "FL-201", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        accepted = [p for p in result["per_passenger"] if p["status"] == "accepted"]
        assert len(accepted) == 1

    def test_deadline_violated_rejected(self):
        """PAX-2 has deadline 14:00. FL-202 arrives at 17:00."""
        ep = _make_episode()
        # Even if SSR were ok, the deadline blocks it. Adjust SSR for this test.
        ep.flights["FL-202"]["supports_ssr"] = ["WCHR"]
        assignments = {"PAX-2": {"flight_id": "FL-202", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        rejected = [p for p in result["per_passenger"] if p["status"] == "rejected"]
        assert len(rejected) == 1
        assert "deadline" in rejected[0]["reason"].lower()

    def test_deadline_met_accepted(self):
        """PAX-2 deadline 14:00, FL-201 arrives 12:00 — OK."""
        ep = _make_episode()
        assignments = {"PAX-2": {"flight_id": "FL-201", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        accepted = [p for p in result["per_passenger"] if p["status"] == "accepted"]
        assert len(accepted) == 1

    def test_availability_decremented(self):
        ep = _make_episode()
        before = ep.flight_availability["FL-201"]["business"]
        assignments = {"PAX-1": {"flight_id": "FL-201", "cabin": "business"}}
        tool_submit_plan(ep, assignments)
        after = ep.flight_availability["FL-201"]["business"]
        assert after == before - 1

    def test_plan_submitted_flag_set(self):
        ep = _make_episode()
        assignments = {"PAX-1": {"flight_id": "FL-201", "cabin": "business"}}
        tool_submit_plan(ep, assignments)
        assert ep.plan_submitted is True

    def test_duplicate_submit_blocked(self):
        ep = _make_episode()
        assignments = {"PAX-1": {"flight_id": "FL-201", "cabin": "business"}}
        tool_submit_plan(ep, assignments)
        result = tool_submit_plan(ep, assignments)
        assert result["status"] == "error"
        assert "already submitted" in result["message"].lower()

    def test_null_assignment_skipped(self):
        ep = _make_episode()
        assignments = {"PAX-1": None, "PAX-3": {"flight_id": "FL-201", "cabin": "economy"}}
        result = tool_submit_plan(ep, assignments)
        skipped = [p for p in result["per_passenger"] if p["status"] == "skipped"]
        assert len(skipped) == 1
        assert result["accepted_count"] == 1

    def test_hard_group_split_violation(self):
        """GRP-1 (hard) split across flights → constraint_violations."""
        ep = _make_episode()
        assignments = {
            "PAX-2": {"flight_id": "FL-201", "cabin": "economy"},
            "PAX-3": {"flight_id": "FL-202", "cabin": "economy"},
        }
        result = tool_submit_plan(ep, assignments)
        assert len(result["constraint_violations"]) > 0
        assert any("GRP-1" in v for v in result["constraint_violations"])

    def test_hard_group_together_no_violation(self):
        """GRP-1 (hard) on same flight → no violations."""
        ep = _make_episode()
        assignments = {
            "PAX-2": {"flight_id": "FL-201", "cabin": "economy"},
            "PAX-3": {"flight_id": "FL-201", "cabin": "economy"},
        }
        result = tool_submit_plan(ep, assignments)
        hard_violations = [
            v for v in result["constraint_violations"]
            if "GRP-1" in v
        ]
        assert len(hard_violations) == 0

    def test_bookings_reset_on_submit(self):
        """submit_plan resets bookings to {} before processing."""
        ep = _make_episode()
        ep.bookings["PAX-1"] = {"flight_id": "FL-201", "cabin": "business"}
        assignments = {"PAX-3": {"flight_id": "FL-201", "cabin": "economy"}}
        tool_submit_plan(ep, assignments)
        # Old booking for PAX-1 should be gone
        assert "PAX-1" not in ep.bookings
        assert "PAX-3" in ep.bookings


# ───────────────────────────────────────────────────────────
# tool_finalize_plan
# ───────────────────────────────────────────────────────────

class TestFinalizePlan:
    def test_sets_done(self):
        ep = _make_episode()
        tool_finalize_plan(ep)
        assert ep.done is True

    def test_without_plan_warns(self):
        ep = _make_episode()
        result = tool_finalize_plan(ep)
        assert result["status"] == "success"
        assert "without" in result["message"].lower()

    def test_with_plan_ok(self):
        ep = _make_episode()
        ep.plan_submitted = True
        result = tool_finalize_plan(ep)
        assert result["status"] == "success"
        assert "finalized" in result["message"].lower()


# ───────────────────────────────────────────────────────────
# VALID_CABINS constant
# ───────────────────────────────────────────────────────────

class TestValidCabins:
    def test_expected_cabins(self):
        assert VALID_CABINS == {"economy", "premium_economy", "business"}
