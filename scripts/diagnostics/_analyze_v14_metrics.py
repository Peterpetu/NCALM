#!/usr/bin/env python3
"""Analyze V14 training metrics."""
import json, sys

lines = open("experiments/AENCA v14 (Emergent Dynamics)/v14_smoke_20min_20260331_124646/metrics.jsonl").readlines()

print(f"Total metric entries: {len(lines)}")
print(f"{'step':>6} {'ce':>8} {'div':>10} {'dyn':>12} {'reg':>8} {'halt_kl':>8}")
print("-" * 60)

for i in [0, 1, 5, 10, 25, 50, 100, len(lines)//4, len(lines)//2, 3*len(lines)//4, len(lines)-1]:
    if i >= len(lines):
        continue
    m = json.loads(lines[i])
    s = m.get("step", "?")
    ce = m.get("ce", m.get("val_ce", ""))
    div = m.get("div", "")
    dyn = m.get("dyn", "")
    reg = m.get("reg", "")
    hk = m.get("halt_kl", "")
    
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v) if v != "" else "  ---"
    
    print(f"{s:>6} {fmt(ce):>8} {fmt(div):>10} {fmt(dyn):>12} {fmt(reg):>8} {fmt(hk):>8}")

# Check if div loss changed much
print("\n--- Routing diversity loss trend ---")
divs = []
dyns = []
for line in lines:
    m = json.loads(line)
    if "div" in m:
        divs.append((m["step"], m["div"]))
    if "dyn" in m:
        dyns.append((m["step"], m["dyn"]))

if divs:
    print(f"  div @ step {divs[0][0]}: {divs[0][1]:.6f}")
    print(f"  div @ step {divs[len(divs)//2][0]}: {divs[len(divs)//2][1]:.6f}")
    print(f"  div @ step {divs[-1][0]}: {divs[-1][1]:.6f}")

if dyns:
    print(f"  dyn @ step {dyns[0][0]}: {dyns[0][1]:.6f}")
    print(f"  dyn @ step {dyns[len(dyns)//2][0]}: {dyns[len(dyns)//2][1]:.6f}")
    print(f"  dyn @ step {dyns[-1][0]}: {dyns[-1][1]:.6f}")

# Check extra metrics for rule norms and routing
print("\n--- Extra metrics (rule norms, router) at end ---")
last = json.loads(lines[-1])
for k, v in sorted(last.items()):
    if any(x in k for x in ["rule", "router", "feedback", "readout", "scale", "content", "ffn"]):
        print(f"  {k}: {v}")
