"""
Calibration Prediction Script v4
Direct distance-based prediction with clear boundaries.
Target: Friday 9th January 2026
"""

import pandas as pd
from datetime import datetime
import json


# Load data
df = pd.read_csv("data/22.csv")
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Target date: Friday 9th January 2026
TARGET_DATE = datetime(2026, 1, 9)

print(f"Target Date: {TARGET_DATE.strftime('%Y-%m-%d')} (Friday)")
print("=" * 60)

# Get last known values (1/5/2026 - most recent Friday in dataset)
last_fri = df[df["Date"] <= "2026-01-05"].iloc[-1]
print(f"Last known data: {last_fri['Date'].strftime('%Y-%m-%d')}")
print(
    f"Positions: P1={last_fri['P1']}, P2={last_fri['P2']}, P3={last_fri['P3']}, P4={last_fri['P4']}, P5={last_fri['P5']}"
)
print(f"Positions: P6={last_fri['P6']}, P7={last_fri['P7']}")

# Direct values from last record
base_p1 = int(last_fri["P1"])
base_p2 = int(last_fri["P2"])
base_p3 = int(last_fri["P3"])
base_p4 = int(last_fri["P4"])
base_p5 = int(last_fri["P5"])
base_p6 = int(last_fri["P6"])
base_p7 = int(last_fri["P7"])

# User's specified distance pattern for P1-P5
# From P1 to P2 is 5, P2 to P3 is 2, P3 to P4 is 11, P4 to P5 is 10
# These are the distances to ADD to get next positions
DIST_P1_P2 = 5
DIST_P2_P3 = 2
DIST_P3_P4 = 11
DIST_P4_P5 = 10

# Calculate next positions by ADDING distances to each base value
p1_next = min(50, base_p1 + DIST_P1_P2)
p2_next = min(50, base_p2 + DIST_P2_P3)
p3_next = min(50, base_p3 + DIST_P3_P4)
p4_next = min(50, base_p4 + DIST_P4_P5)
p5_next = min(50, base_p5 + DIST_P4_P5)  # Use same as P4->P5

# For P6-P7, add distance (12-10=2)
p6_next = base_p6
p7_next = min(12, base_p6 + 2)

# Create ordered, unique positions
all_p1_5 = [p1_next, p2_next, p3_next, p4_next, p5_next]
most_likely_p1_5 = sorted(set(all_p1_5))
while len(most_likely_p1_5) < 5:
    # Add a number in range that doesn't exist yet
    for n in range(1, 51):
        if n not in most_likely_p1_5:
            most_likely_p1_5.append(n)
            break
most_likely_p1_5 = sorted(most_likely_p1_5)[:5]

all_p6_7 = [p6_next, p7_next]
most_likely_p6_7 = sorted(set(all_p6_7))
while len(most_likely_p6_7) < 2:
    for n in range(1, 13):
        if n not in most_likely_p6_7:
            most_likely_p6_7.append(n)
            break
most_likely_p6_7 = sorted(most_likely_p6_7)[:2]

print(f"\nDirect calculation with distances:")
print(f"P1: {base_p1} + {DIST_P1_P2} = {p1_next}")
print(f"P2: {base_p2} + {DIST_P2_P3} = {p2_next}")
print(f"P3: {base_p3} + {DIST_P3_P4} = {p3_next}")
print(f"P4: {base_p4} + {DIST_P4_P5} = {p4_next}")
print(f"P5: {base_p5} + {DIST_P4_P5} = {p5_next}")
print(f"P6: {base_p6} + 0 = {p6_next}")
print(f"P7: {base_p6} + 2 = {p7_next}")

# For LEAST LIKELY, use SUBTRACTION of distances (opposite pattern)
# This creates clear boundary from most likely
least_p1 = max(1, base_p1 - DIST_P1_P2)
least_p2 = max(1, base_p2 - DIST_P2_P3)
least_p3 = max(1, base_p3 - DIST_P3_P4)
least_p4 = max(1, base_p4 - DIST_P4_P5)
least_p5 = max(1, base_p5 - DIST_P4_P5)

least_likely_p1_5 = sorted(set([least_p1, least_p2, least_p3, least_p4, least_p5]))[:5]

least_likely_p6_7 = sorted(set([max(1, base_p6 - 2), max(1, base_p7 - 2)]))[:2]

print(f"\nOpposite calculation (subtraction):")
print(f"P1: {base_p1} - {DIST_P1_P2} = {least_p1}")
print(f"P2: {base_p2} - {DIST_P2_P3} = {least_p2}")
print(f"P3: {base_p3} - {DIST_P3_P4} = {least_p3}")
print(f"P4: {base_p4} - {DIST_P4_P5} = {least_p4}")
print(f"P5: {base_p5} - {DIST_P4_P5} = {least_p5}")
print(f"Least P6: {max(1, base_p6 - 2)}, Least P7: {max(1, base_p7 - 2)}")

# Calculate distances
dist_most_p1_5 = [most_likely_p1_5[i + 1] - most_likely_p1_5[i] for i in range(4)]
dist_most_p6_7 = [most_likely_p6_7[1] - most_likely_p6_7[0]]

dist_least_p1_5 = [least_likely_p1_5[i + 1] - least_likely_p1_5[i] for i in range(4)]
dist_least_p6_7 = [least_likely_p6_7[1] - least_likely_p6_7[0]]

print(f"\nMOST LIKELY: P1-P5={most_likely_p1_5}, P6-P7={most_likely_p6_7}")
print(f"LEAST LIKELY: P1-P5={least_likely_p1_5}, P6-P7={least_likely_p6_7}")

# Final result
result = {
    "target_date": TARGET_DATE.strftime("%Y-%m-%d"),
    "day_of_week": "Friday",
    "most_likely": {
        "positions": {
            "P1": most_likely_p1_5[0],
            "P2": most_likely_p1_5[1],
            "P3": most_likely_p1_5[2],
            "P4": most_likely_p1_5[3],
            "P5": most_likely_p1_5[4],
            "P6": most_likely_p6_7[0],
            "P7": most_likely_p6_7[1],
        },
        "distances": {
            "P1_P2": dist_most_p1_5[0],
            "P2_P3": dist_most_p1_5[1],
            "P3_P4": dist_most_p1_5[2],
            "P4_P5": dist_most_p1_5[3],
            "P6_P7": dist_most_p6_7[0],
        },
    },
    "least_likely": {
        "positions": {
            "P1": least_likely_p1_5[0],
            "P2": least_likely_p1_5[1],
            "P3": least_likely_p1_5[2],
            "P4": least_likely_p1_5[3],
            "P5": least_likely_p1_5[4],
            "P6": least_likely_p6_7[0],
            "P7": least_likely_p6_7[1],
        },
        "distances": {
            "P1_P2": dist_least_p1_5[0],
            "P2_P3": dist_least_p1_5[1],
            "P3_P4": dist_least_p1_5[2],
            "P4_P5": dist_least_p1_5[3],
            "P6_P7": dist_least_p6_7[0],
        },
    },
}

print("\n" + "=" * 60)
print("FINAL PREDICTION FOR FRIDAY 9TH JANUARY 2026")
print("=" * 60)
print(json.dumps(result, indent=2))
