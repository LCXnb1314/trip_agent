import json

path = "/home/data/lcxnb1314/big/hello-agents-main/my_agent/logs/2026-08-13/32631a3ab211.jsonl"

with open(
    path,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        record = json.loads(line)

        print(
            record["node"],
            record["event"]
        )

import pandas as pd

df = pd.read_json(
    path,
    lines=True
)

print(df)