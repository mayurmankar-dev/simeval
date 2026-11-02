"""Compare two benchmark runs on the SAME scenes (paired comparison).

    python compare.py results/v1_seed0.json results/v2_seed0.json
"""
import json, sys
from collections import Counter
from math import comb
from benchmark import wilson_ci


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test: are b (fixed) and c (broken) different beyond chance?"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


a_path, b_path = sys.argv[1], sys.argv[2]
A = {r["scene_id"]: r for r in json.loads(open(a_path).read())}
B = {r["scene_id"]: r for r in json.loads(open(b_path).read())}
ids = sorted(set(A) & set(B))

for name, R in [("before", A), ("after", B)]:
    k = sum(R[i]["success"] for i in ids); lo, hi = wilson_ci(k, len(ids))
    print(f"{name:7s} {k}/{len(ids)} = {k/len(ids):.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

fixed = [i for i in ids if not A[i]["success"] and B[i]["success"]]
broke = [i for i in ids if A[i]["success"] and not B[i]["success"]]
print(f"\nPaired: {len(fixed)} scenes fixed, {len(broke)} scenes broken, "
      f"{len(ids) - len(fixed) - len(broke)} unchanged")
print(f"McNemar exact p = {mcnemar_exact(len(fixed), len(broke)):.2g}  (small p = real change, not chance)")

print("\nOutcome counts (before -> after):")
ca, cb = Counter(A[i]["outcome"] for i in ids), Counter(B[i]["outcome"] for i in ids)
for o in sorted(set(ca) | set(cb), key=lambda o: -ca.get(o, 0)):
    print(f"  {o:16s} {ca.get(o, 0):3d} -> {cb.get(o, 0):3d}")
if broke:
    print("\nNewly failing scenes (check these!):", [(i, B[i]["outcome"]) for i in broke])