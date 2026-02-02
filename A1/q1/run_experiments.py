
import sys
import subprocess
import math
import os
import time

apriori_path = sys.argv[1]
fp_path = sys.argv[2]
dataset = sys.argv[3]
outdir = sys.argv[4]

supports = [5, 10, 25, 50, 90]

with open(dataset, "r") as f:
    num_transactions = sum(1 for _ in f)

results = []

for s in supports:
    minsup = math.ceil((s / 100) * num_transactions)

    ap_out = os.path.join(outdir, f"ap{s}")
    start = time.time()
    subprocess.run(
        [apriori_path, f"-s{-minsup}", dataset, ap_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    ap_time = time.time() - start

    fp_out = os.path.join(outdir, f"fp{s}")
    start = time.time()
    subprocess.run(
        [fp_path, f"-s{-minsup}", dataset, fp_out],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    fp_time = time.time() - start

    results.append((s, ap_time, fp_time))

with open(os.path.join(outdir, "runtimes.csv"), "w") as f:
    f.write("support,apriori,fpgrowth\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]}\n")