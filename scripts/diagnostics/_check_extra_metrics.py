#!/usr/bin/env python3
import json
lines = open("experiments/AENCA v14 (Emergent Dynamics)/v14_smoke_20min_20260331_124646/metrics.jsonl").readlines()
last_train = None
for line in lines:
    m = json.loads(line)
    if "morph_rule0_norm" in m:
        last_train = m
if last_train:
    for k, v in sorted(last_train.items()):
        if any(x in k for x in ["rule", "router", "feedback", "readout", "scale", "content", "ffn"]):
            print(f"  {k}: {v}")
else:
    print("No extra metrics found")
    m = json.loads(lines[-1])
    print("Keys in last entry:", sorted(m.keys()))
