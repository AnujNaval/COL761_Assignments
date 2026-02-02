import sys
import csv
import matplotlib.pyplot as plt
import os

outdir = sys.argv[1]

supports = []
ap_times = []
fp_times = []

with open(os.path.join(outdir, "runtimes.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        supports.append(int(row["support"]))
        ap_times.append(float(row["apriori"]))
        fp_times.append(float(row["fpgrowth"]))

plt.figure()
plt.plot(supports, ap_times, marker='o', label="Apriori")
plt.plot(supports, fp_times, marker='o', label="FP-Growth")
plt.xlabel("Minimum Support (%)")
plt.ylabel("Runtime (seconds)")
plt.legend()
plt.savefig(os.path.join(outdir, "plot.png"))