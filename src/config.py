from datetime import date

TARGET_DATE = date(2026, 1, 2)
DAY_OF_WEEK = 4  # Friday
P1_5_RANGE = (1, 50)
P6_7_RANGE = (1, 12)

# Last prediction and actual for error correction
LAST_PREDICTION = [12, 33, 42, 45, 47, 8, 11]
LAST_ACTUAL = [4, 16, 40, 41, 44, 2, 10]

# Refined boundaries based on last actual
P1_5_REFINED_RANGE = (4, 44)  # min and max of last actual for P1-P5
P6_7_REFINED_RANGE = (2, 10)  # min and max of last actual for P6-P7
