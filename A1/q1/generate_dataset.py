import sys
import random

U = int(sys.argv[1])          
N = int(sys.argv[2])          

random.seed(42)

items = list(range(1, U + 1))

num_high = int(0.30 * U)
num_mid  = int(0.45 * U)

high_freq = items[:num_high]
mid_freq  = items[num_high:num_high + num_mid]
low_freq  = items[num_high + num_mid:]

transactions = []

for _ in range(N):
    txn = set()

    for item in high_freq:
        if random.random() < 0.5:
            txn.add(item)

    for item in random.sample(mid_freq, k=random.randint(5, 15)):
        if random.random() < 0.25:
            txn.add(item)

    for item in random.sample(low_freq, k=random.randint(3, 10)):
        if random.random() < 0.05:
            txn.add(item)

    if not txn:
        txn.add(random.choice(items))

    transactions.append(sorted(txn))

with open("generated_transactions.dat", "w") as f:
    for txn in transactions:
        f.write(" ".join(map(str, txn)) + "\n")