import json
from load import load_data
from features import distance_features
from calibration import calibrate_p1_5, calibrate_p6_7
from collections import Counter
from llm_agent import llm_refine

df = load_data("data/22.csv")
df = distance_features(df)

# Define lines
lines_p1_5 = {
    1: list(range(1,10)),
    2: list(range(10,19)),
    3: list(range(19,28)),
    4: list(range(28,37)),
    5: list(range(37,46)),
    6: list(range(46,51))
}

lines_p6_7 = {
    1: list(range(1,5)),
    2: list(range(5,9)),
    3: list(range(9,13))
}

# Find busy line for P1-P5
line_counts = {i: 0 for i in range(1,7)}
for _, row in df.iterrows():
    for p in ['P1','P2','P3','P4','P5']:
        pos = int(row[p])
        for line, rng in lines_p1_5.items():
            if pos in rng:
                line_counts[line] += 1
busy_line_p1_5 = max(line_counts, key=line_counts.get)

# Find busy line for P6-P7
line_counts_p6_7 = {i: 0 for i in range(1,4)}
for _, row in df.iterrows():
    for p in ['P6','P7']:
        pos = int(row[p])
        for line, rng in lines_p6_7.items():
            if pos in rng:
                line_counts_p6_7[line] += 1
busy_line_p6_7 = max(line_counts_p6_7, key=line_counts_p6_7.get)

# Exact calculation of boundaries and increment epochs for x,y,z locations (deep abstract mathematics)
epochs_data = {}
for p in ['P1','P2','P3','P4','P5','P6','P7']:
    values = df[p].values
    increments = [values[i+1] - values[i] for i in range(len(values)-1)]
    boundaries = [int(values.min()), int(values.max())]
    last_increment = increments[-1] if increments else 0
    next_value = values[-1] + last_increment
    # clamp to boundaries
    next_value = max(boundaries[0], min(boundaries[1], next_value))
    epochs_data[p] = {
        'values': values.tolist(),
        'increments': increments,
        'boundaries': boundaries,
        'next': int(next_value)
    }

# Most likely: exact next positions based on increment epochs
next_p1_5 = sorted([epochs_data[p]['next'] for p in ['P1','P2','P3','P4','P5']])
most_likely_p1_5 = next_p1_5
most_likely_p6_7 = [epochs_data['P6']['next'], epochs_data['P7']['next']]

# Least likely: least common in respective ranges
ranges_p = {
    'P1': range(1,10),
    'P2': range(10,19),
    'P3': range(19,28),
    'P4': range(28,37),
    'P5': range(37,46),
    'P6': range(1,5),
    'P7': range(5,9)
}

least_likely = {}
for p, rng in ranges_p.items():
    counts = df[p].value_counts()
    candidates = counts[counts.index.isin(rng)]
    if not candidates.empty:
        least_likely[p] = int(candidates.idxmin())
    else:
        least_likely[p] = min(rng)

# Calculate distances
def calc_distances(positions):
    return [positions[i+1] - positions[i] for i in range(len(positions)-1)]

dist_p1_5 = calc_distances(most_likely_p1_5)
dist_p6_7 = calc_distances(most_likely_p6_7)

least_likely_p1_5 = [least_likely['P1'], least_likely['P2'], least_likely['P3'], least_likely['P4'], least_likely['P5']]
least_likely_p6_7 = [least_likely['P6'], least_likely['P7']]
dist_least_p1_5 = calc_distances(least_likely_p1_5)
dist_least_p6_7 = calc_distances(least_likely_p6_7)

result = {
    "most_likely": {
        "positions": {
            "P1": most_likely_p1_5[0],
            "P2": most_likely_p1_5[1],
            "P3": most_likely_p1_5[2],
            "P4": most_likely_p1_5[3],
            "P5": most_likely_p1_5[4],
            "P6": most_likely_p6_7[0],
            "P7": most_likely_p6_7[1]
        },
        "distances": {
            "P1_P5": dist_p1_5,
            "P6_P7": dist_p6_7
        }
    },
    "least_likely": {
        "positions": least_likely,
        "distances": {
            "P1_P5": dist_least_p1_5,
            "P6_P7": dist_least_p6_7
        }
    }
}

# Refine using deep abstract mathematics (LLM) - commented out due to API key
# result = llm_refine(result)

print(json.dumps(result))
