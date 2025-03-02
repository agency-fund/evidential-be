#!/usr/bin/env python
"""parse output of /assign and re-output as csv"""

import argparse
import csv
import json
import sys
# import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input file. If not supplied, uses stdin")
parser.add_argument("-o", help="output file")
args = parser.parse_args()

if args.i:
    with open(args.i, encoding="utf-8") as f:
        data = json.load(f)
else:
    print("Reading from stdin...", file=sys.stderr)
    data = json.load(sys.stdin)

assignments = data["assignments"]

# Create header from first record.
# May be less brittle if we use a pandas in case not all records have all strata
strata = [s["field_name"] for s in assignments[0]["strata"]]
headers = ["participant_id", "arm_name", *strata]

rows = []
for assignment in assignments:
    row = {
        "participant_id": assignment["participant_id"],
        "arm_name": assignment["arm_name"],
    }
    for s in assignment["strata"]:
        row[s["field_name"]] = s["strata_value"]
    rows.append(row)

# df = pd.DataFrame(rows)[headers]
# df.to_csv('assignments.csv', index=False)

if args.o:
    with open(args.o, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for assignment in rows:
            writer.writerow(list(assignment.values()))
else:
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    for assignment in rows:
        writer.writerow(list(assignment.values()))
