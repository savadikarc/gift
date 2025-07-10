import json
import argparse

parser = argparse.ArgumentParser(description='Get score from json file')
parser.add_argument('--file', type=str, help='jsonl file')
args = parser.parse_args()

# Load jsonl file
with open(args.file, 'r') as f:
    data = f.readlines()

scores = []
for line in data:
    line = json.loads(line)
    if "multi-turn" in line["judge"][1]:
        continue
    # print(line["question_id"], line["score"])
    scores.append(line["score"])

print(len(scores), sum(scores) / len(scores))
