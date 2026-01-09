"""
Calibration Prediction Script v3
Direct distance-based prediction with minimal statistics.
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

# User's specified distance pattern for P1-P5
# From P1 to P2 is 5, P2 to P3 is 2, P3 to P4 is 11, P4 to P5 is 10
DIST_P1_P2 = 5
DIST_P2_P3 = 2
DIST_P3_P4 = 11
DIST_P4_P5 = 10

# Direct calculation using last known values
base_p1 = int(last_fri["P1"])
base_p2 = int(last_fri["P2"])
base_p3 = int(last_fri["P3"])
base_p4 = int(last_fri["P4"])
base_p5 = int(last_fri["P5"])
base_p6 = int(last_fri["P6"])
base_p7 = int(last_fri["P7"])

# Calculate next positions using user's specified distances
# Starting from the last known positions
p1_pred = base_p1 + DIST_P1_P2
p2_pred = base_p2 + DIST_P2_P3
p3_pred = base_p3 + DIST_P3_P4
p4_pred = base_p4 + DIST_P4_P5

# For P6-P7, use the actual distance from last record (12 - 10 = 2)
actual_d67 = base_p7 - base_p6
p6_pred = base_p6
p7_pred = base_p6 + actual_d67

# Ensure within range and sort
most_likely_p1_5 = sorted(set([p1_pred, p2_pred, p3_pred, p4_pred, base_p5]))
while len(most_likely_p1_5) < 5:
    missing = min(set(range(1, 51)) - set(most_likely_p1_5))
    most_likely_p1_5.append(missing)
most_likely_p1_5 = sorted(most_likely_p1_5)[:5]

most_likely_p6_7 = sorted(set([p6_pred, p7_pred]))
while len(most_likely_p6_7) < 2:
    missing = min(set(range(1, 13)) - set(most_likely_p6_7))
    most_likely_p6_7.append(missing)
most_likely_p6_7 = sorted(most_likely_p6_7)[:2]

# Least likely: positions not commonly appearing together
# Use the pattern from actual last record distances
actual_dist_p1_p2 = base_p2 - base_p1  # 9
actual_dist_p2_p3 = base_p3 - base_p2  # 3
actual_dist_p3_p4 = base_p4 - base_p3  # 1
actual_dist_p4_p5 = base_p5 - base_p4  # 13

# Build least likely using inverse pattern
least_p1 = max(1, min(50, base_p1 - 5))
least_p2 = max(1, min(50, base_p2 - 3))
least_p3 = max(1, min(50, base_p3 - 1))
least_p4 = max(1, min(50, base_p4 - 13))
least_p5 = max(1, min(50, base_p5 - 5))

least_likely_p1_5 = sorted(set([least_p1, least_p2, least_p3, least_p4, least_p5]))[:5]

least_likely_p6_7 = sorted([4, 1])  # Based on historical least frequent

# Calculate distances
dist_most_p1_5 = [most_likely_p1_5[i + 1] - most_likely_p1_5[i] for i in range(4)]
dist_most_p6_7 = [most_likely_p6_7[1] - most_likely_p6_7[0]]

dist_least_p1_5 = [least_likely_p1_5[i + 1] - least_likely_p1_5[i] for i in range(4)]
dist_least_p6_7 = [least_likely_p6_7[1] - least_likely_p6_7[0]]

print("\n" + "=" * 60)
print("DIRECT DISTANCE-BASED PREDICTION")
print("=" * 60)
print(
    f"Using distances: P1->P2={DIST_P1_P2}, P2->P3={DIST_P2_P3}, P3->P4={DIST_P3_P4}, P4->P5={DIST_P4_P5}"
)
print(
    f"Last actual distances: P1->P2={actual_dist_p1_p2}, P2->P3={actual_dist_p2_p3}, P3->P4={actual_dist_p3_p4}, P4->P5={actual_dist_p4_p5}"
)
print(f"Last P6->P7 distance: {actual_d67}")

print(f"\nMOST LIKELY positions: P1-P5={most_likely_p1_5}, P6-P7={most_likely_p6_7}")
print(f"LEAST LIKELY positions: P1-P5={least_likely_p1_5}, P6-P7={least_likely_p6_7}")

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
