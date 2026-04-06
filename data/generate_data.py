"""
Generate mock data for the Airline Seat Reassignment Environment.

Outputs:
  - ac1_config.json   : AC-1 seat configuration
  - ac2_config.json   : AC-2 seat configuration
  - passengers.json   : Passenger list + initial assignment dicts
  - seats_ac1.csv     : AC-1 seats (pandas-friendly)
  - seats_ac2.csv     : AC-2 seats (pandas-friendly)
  - passengers.csv    : Passenger records (pandas-friendly)
  - assignments.csv   : Current seat assignments (ac1 fully occupied, ac2 empty)
"""

import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
random.seed(SEED)

DATA_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Seat type mappings
# ---------------------------------------------------------------------------
AC1_BUSINESS_TYPES = {"A": "window", "B": "aisle", "C": "aisle", "D": "window"}
AC1_ECONOMY_TYPES  = {"A": "window", "B": "middle", "C": "aisle",
                      "D": "aisle",  "E": "middle", "F": "window"}

AC2_BUSINESS_TYPES = {"A": "window", "B": "aisle", "C": "aisle", "D": "window"}
AC2_ECONOMY_TYPES  = {"A": "window", "B": "aisle",  "C": "middle", "D": "aisle",
                      "E": "aisle",  "F": "middle", "G": "aisle",  "H": "window"}


def build_seats(business_rows, business_cols, business_types,
                economy_rows, economy_cols, economy_types):
    seats = []
    for row in business_rows:
        for col in business_cols:
            seats.append({
                "seat_id": f"{row}{col}",
                "cabin": "business",
                "seat_type": business_types[col],
            })
    for row in economy_rows:
        for col in economy_cols:
            seats.append({
                "seat_id": f"{row}{col}",
                "cabin": "economy",
                "seat_type": economy_types[col],
            })
    return seats


# ---------------------------------------------------------------------------
# Aircraft configurations
# ---------------------------------------------------------------------------
ac1_seats = build_seats(
    business_rows=[1, 2], business_cols=["A", "B", "C", "D"],
    business_types=AC1_BUSINESS_TYPES,
    economy_rows=[3, 4],  economy_cols=["A", "B", "C", "D", "E", "F"],
    economy_types=AC1_ECONOMY_TYPES,
)

ac2_seats = build_seats(
    business_rows=[1, 2], business_cols=["A", "B", "C", "D"],
    business_types=AC2_BUSINESS_TYPES,
    economy_rows=[3, 4],  economy_cols=["A", "B", "C", "D", "E", "F", "G", "H"],
    economy_types=AC2_ECONOMY_TYPES,
)

ac1_config = {
    "aircraft_id": "AC-1",
    "layout": {
        "business": {"rows": [1, 2], "columns": ["A", "B", "C", "D"], "pattern": "2-2"},
        "economy":  {"rows": [3, 4], "columns": ["A", "B", "C", "D", "E", "F"], "pattern": "3-3"},
    },
    "seats": ac1_seats,
}

ac2_config = {
    "aircraft_id": "AC-2",
    "layout": {
        "business": {"rows": [1, 2], "columns": ["A", "B", "C", "D"], "pattern": "1-2-1"},
        "economy":  {"rows": [3, 4], "columns": ["A", "B", "C", "D", "E", "F", "G", "H"], "pattern": "2-4-2"},
    },
    "seats": ac2_seats,
}

# ---------------------------------------------------------------------------
# Passenger generation
# ---------------------------------------------------------------------------
NAMES = [
    "Aarav Sharma",  "Priya Patel",   "Rohan Gupta",   "Ananya Singh",
    "Vikram Mehta",  "Deepa Nair",    "Arjun Kumar",   "Shreya Iyer",
    "Rahul Verma",   "Kavya Reddy",   "Aditya Joshi",  "Pooja Agarwal",
    "Kiran Rao",     "Neha Mishra",   "Suresh Pillai", "Riya Desai",
    "Arun Pandey",   "Divya Chopra",  "Manish Kapoor", "Sana Bose",
]

# Every AC-1 seat is occupied; assign passengers in seat order
ac1_seat_ids = [s["seat_id"] for s in ac1_seats]  # 20 seats
assert len(ac1_seat_ids) == 20 and len(NAMES) == 20

# Identify window seats on AC-1 for paid_window eligibility
ac1_window_seats = {s["seat_id"] for s in ac1_seats if s["seat_type"] == "window"}
# Business windows: 1A,1D,2A,2D  |  Economy windows: 3A,3F,4A,4F

# Deterministically choose paid_window passengers:
# 2 business windows (1A→PAX-001, 1D→PAX-004) + 3 economy windows (3A→PAX-009, 3F→PAX-014, 4A→PAX-015)
PAID_WINDOW_SEATS = {"1A", "1D", "3A", "3F", "4A"}

passengers = []
ac1_assignments = {}

for i, (seat_id, name) in enumerate(zip(ac1_seat_ids, NAMES)):
    pax_id = f"PAX-{i+1:03d}"
    seat_info = next(s for s in ac1_seats if s["seat_id"] == seat_id)
    paid_window = seat_id in PAID_WINDOW_SEATS
    passengers.append({
        "passenger_id": pax_id,
        "name": name,
        "seat_ac1": seat_id,
        "cabin": seat_info["cabin"],
        "paid_window": paid_window,
    })
    ac1_assignments[seat_id] = pax_id

passengers_output = {
    "passengers": passengers,
    "ac1_assignments": ac1_assignments,
    "ac2_assignments": {},
}

# ---------------------------------------------------------------------------
# Write JSON files
# ---------------------------------------------------------------------------
(DATA_DIR / "ac1_config.json").write_text(json.dumps(ac1_config, indent=2))
(DATA_DIR / "ac2_config.json").write_text(json.dumps(ac2_config, indent=2))
(DATA_DIR / "passengers.json").write_text(json.dumps(passengers_output, indent=2))
print("JSON files written.")

# ---------------------------------------------------------------------------
# Write CSV files (pandas)
# ---------------------------------------------------------------------------
# seats_ac1.csv
df_ac1 = pd.DataFrame(ac1_seats)
df_ac1.insert(0, "aircraft_id", "AC-1")
df_ac1.to_csv(DATA_DIR / "seats_ac1.csv", index=False)

# seats_ac2.csv
df_ac2 = pd.DataFrame(ac2_seats)
df_ac2.insert(0, "aircraft_id", "AC-2")
df_ac2.to_csv(DATA_DIR / "seats_ac2.csv", index=False)

# passengers.csv
df_pax = pd.DataFrame(passengers)
df_pax.to_csv(DATA_DIR / "passengers.csv", index=False)

# assignments.csv — one row per AC-1 seat; ac2_seat starts empty
df_assign = pd.DataFrame([
    {"passenger_id": pid, "seat_ac1": seat, "seat_ac2": None}
    for seat, pid in ac1_assignments.items()
])
df_assign.to_csv(DATA_DIR / "assignments.csv", index=False)

print("CSV files written.")
print(f"\nPaid-window passengers ({df_pax['paid_window'].sum()} total):")
print(df_pax[df_pax["paid_window"]][["passenger_id", "name", "seat_ac1", "cabin"]])
