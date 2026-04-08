"""
Generate mock data for the Hard Airline Seat Reassignment task.

Differences from medium:
  - seats_ac1/seats_ac2 gain an `extra_legroom` column
  - passengers gain a `paid_legroom` column
  - AC-1: business row 1 has legroom; economy row 3 has legroom
  - AC-2: business row 2 has legroom (SHIFTED from AC-1 row 1);
          economy seats 4A, 4B, 4H have legroom (exactly 3 for 3 paid passengers)
  - Constraint summary (20 passengers):
      paid_window  : PAX-001, PAX-004, PAX-008, PAX-009, PAX-014, PAX-015  (6 total)
      paid_legroom : PAX-001, PAX-002, PAX-005, PAX-009, PAX-010, PAX-012  (6 total)
      paid_both    : PAX-001, PAX-009  (2 — most constrained)

Max grader_score = 1.0 (all constraints satisfiable with optimal planning).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Seat definitions
# ---------------------------------------------------------------------------
AC1_SEATS = [
    {"seat_id": "1A", "cabin": "business", "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "1B", "cabin": "business", "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "1C", "cabin": "business", "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "1D", "cabin": "business", "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "2A", "cabin": "business", "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "2B", "cabin": "business", "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "2C", "cabin": "business", "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "2D", "cabin": "business", "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "3A", "cabin": "economy",  "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "3B", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": True},
    {"seat_id": "3C", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "3D", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "3E", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": True},
    {"seat_id": "3F", "cabin": "economy",  "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "4A", "cabin": "economy",  "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "4B", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "4C", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "4D", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "4E", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "4F", "cabin": "economy",  "seat_type": "window",  "extra_legroom": False},
]

AC2_SEATS = [
    {"seat_id": "1A", "cabin": "business", "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "1B", "cabin": "business", "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "1C", "cabin": "business", "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "1D", "cabin": "business", "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "2A", "cabin": "business", "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "2B", "cabin": "business", "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "2C", "cabin": "business", "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "2D", "cabin": "business", "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "3A", "cabin": "economy",  "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "3B", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "3C", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "3D", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "3E", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "3F", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "3G", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "3H", "cabin": "economy",  "seat_type": "window",  "extra_legroom": False},
    {"seat_id": "4A", "cabin": "economy",  "seat_type": "window",  "extra_legroom": True},
    {"seat_id": "4B", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": True},
    {"seat_id": "4C", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "4D", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "4E", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "4F", "cabin": "economy",  "seat_type": "middle",  "extra_legroom": False},
    {"seat_id": "4G", "cabin": "economy",  "seat_type": "aisle",   "extra_legroom": False},
    {"seat_id": "4H", "cabin": "economy",  "seat_type": "window",  "extra_legroom": True},
]

PASSENGERS = [
    {"passenger_id": "PAX-001", "name": "Aarav Sharma",  "seat_ac1": "1A", "cabin": "business", "paid_window": True,  "paid_legroom": True},
    {"passenger_id": "PAX-002", "name": "Priya Patel",   "seat_ac1": "1B", "cabin": "business", "paid_window": False, "paid_legroom": True},
    {"passenger_id": "PAX-003", "name": "Rohan Gupta",   "seat_ac1": "1C", "cabin": "business", "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-004", "name": "Ananya Singh",  "seat_ac1": "1D", "cabin": "business", "paid_window": True,  "paid_legroom": False},
    {"passenger_id": "PAX-005", "name": "Vikram Mehta",  "seat_ac1": "2A", "cabin": "business", "paid_window": False, "paid_legroom": True},
    {"passenger_id": "PAX-006", "name": "Deepa Nair",    "seat_ac1": "2B", "cabin": "business", "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-007", "name": "Arjun Kumar",   "seat_ac1": "2C", "cabin": "business", "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-008", "name": "Shreya Iyer",   "seat_ac1": "2D", "cabin": "business", "paid_window": True,  "paid_legroom": False},
    {"passenger_id": "PAX-009", "name": "Rahul Verma",   "seat_ac1": "3A", "cabin": "economy",  "paid_window": True,  "paid_legroom": True},
    {"passenger_id": "PAX-010", "name": "Kavya Reddy",   "seat_ac1": "3B", "cabin": "economy",  "paid_window": False, "paid_legroom": True},
    {"passenger_id": "PAX-011", "name": "Aditya Joshi",  "seat_ac1": "3C", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-012", "name": "Pooja Agarwal", "seat_ac1": "3D", "cabin": "economy",  "paid_window": False, "paid_legroom": True},
    {"passenger_id": "PAX-013", "name": "Kiran Rao",     "seat_ac1": "3E", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-014", "name": "Neha Mishra",   "seat_ac1": "3F", "cabin": "economy",  "paid_window": True,  "paid_legroom": False},
    {"passenger_id": "PAX-015", "name": "Suresh Pillai", "seat_ac1": "4A", "cabin": "economy",  "paid_window": True,  "paid_legroom": False},
    {"passenger_id": "PAX-016", "name": "Riya Desai",    "seat_ac1": "4B", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-017", "name": "Arun Pandey",   "seat_ac1": "4C", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-018", "name": "Divya Chopra",  "seat_ac1": "4D", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-019", "name": "Manish Kapoor", "seat_ac1": "4E", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
    {"passenger_id": "PAX-020", "name": "Sana Bose",     "seat_ac1": "4F", "cabin": "economy",  "paid_window": False, "paid_legroom": False},
]

# Optimal assignment that achieves grader_score = 1.0:
# Business paid_window+legroom (PAX-001) → 2A or 2D (window+legroom)
# Business paid_legroom only (PAX-002, PAX-005) → 2B or 2C (legroom, not window)
# Business paid_window only (PAX-004, PAX-008) → 1A or 1D (window, no legroom)
# Economy paid_window+legroom (PAX-009) → 4A or 4H (window+legroom)
# Economy paid_legroom only (PAX-010, PAX-012) → 4B + remaining 4A/4H
# Economy paid_window only (PAX-014, PAX-015) → 3A or 3H (window, no legroom)

if __name__ == "__main__":
    import csv, json
    from pathlib import Path

    # Write CSVs
    with open(DATA_DIR / "seats_ac1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aircraft_id", "seat_id", "cabin", "seat_type", "extra_legroom"])
        w.writeheader()
        for s in AC1_SEATS:
            w.writerow({"aircraft_id": "AC-1", **s})

    with open(DATA_DIR / "seats_ac2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aircraft_id", "seat_id", "cabin", "seat_type", "extra_legroom"])
        w.writeheader()
        for s in AC2_SEATS:
            w.writerow({"aircraft_id": "AC-2", **s})

    with open(DATA_DIR / "passengers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["passenger_id", "name", "seat_ac1", "cabin", "paid_window", "paid_legroom"])
        w.writeheader()
        w.writerows(PASSENGERS)

    with open(DATA_DIR / "assignments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["passenger_id", "seat_ac1", "seat_ac2"])
        w.writeheader()
        for p in PASSENGERS:
            w.writerow({"passenger_id": p["passenger_id"], "seat_ac1": p["seat_ac1"], "seat_ac2": ""})

    print("Hard task data written.")
    paid_window_count  = sum(1 for p in PASSENGERS if p["paid_window"])
    paid_legroom_count = sum(1 for p in PASSENGERS if p["paid_legroom"])
    paid_both_count    = sum(1 for p in PASSENGERS if p["paid_window"] and p["paid_legroom"])
    print(f"paid_window={paid_window_count}, paid_legroom={paid_legroom_count}, paid_both={paid_both_count}")
