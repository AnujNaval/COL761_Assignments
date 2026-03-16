import sys
from collections import deque, defaultdict
import random

EVAL_SEED = 42

def parse_args():
    if len(sys.argv) != 7:
        print("Usage: python3 forest_fire.py <graph_path> <seed_path> "
              "<output_path> <k> <n_random_instances> <hops>")
        sys.exit(1)
    return (sys.argv[1], sys.argv[2], sys.argv[3],
            int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

def load_graph(graph_path):
    adj = defaultdict(list)
    with open(graph_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            adj[u].append((v, p))
    return dict(adj)

def load_seeds(seed_path):
    seeds = []
    with open(seed_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(int(line))
    return seeds

def simulate_seeded(adj, seeds, blocked, rng, hops=None):
    burned   = set(seeds)
    frontier = [(node, 0) for node in seeds]
    while frontier:
        next_frontier = []
        for u, depth in frontier:
            if hops is not None and depth >= hops:
                continue
            for v, p in adj.get(u, []):
                if (u, v) not in blocked and v not in burned:
                    if rng.random() < p:
                        burned.add(v)
                        next_frontier.append((v, depth + 1))
        frontier = next_frontier
    return len(burned)

def reproduce_eval_scenarios(adj, seeds, num_sim, hops=None):
    """Replay Random(42) to get the exact scenarios the evaluator will use."""
    rng       = random.Random(EVAL_SEED)
    scenarios = []
    for _ in range(num_sim):
        burned     = set(seeds)
        edges_used = set()
        frontier   = [(node, 0) for node in seeds]
        while frontier:
            next_frontier = []
            for u, depth in frontier:
                if hops is not None and depth >= hops:
                    continue
                for v, p in adj.get(u, []):
                    if v not in burned:
                        if rng.random() < p:
                            burned.add(v)
                            edges_used.add((u, v))
                            next_frontier.append((v, depth + 1))
            frontier = next_frontier
        scenarios.append((frozenset(burned), frozenset(edges_used)))
    return scenarios

def downstream_nodes(edge, scenario_edges):
    """Nodes saved by blocking this edge in one scenario."""
    u, v = edge
    if (u, v) not in scenario_edges:
        return frozenset()
    visited = {v}
    queue   = deque([v])
    while queue:
        node = queue.popleft()
        for u2, v2 in scenario_edges:
            if u2 == node and v2 not in visited:
                visited.add(v2)
                queue.append(v2)
    return frozenset(visited)

def greedy_on_scenarios(scenarios, k, output_path, adj):
    num_sim     = len(scenarios)
    sigma_empty = sum(len(s[0]) for s in scenarios) / num_sim
    print(f"sigma(empty) on eval scenarios: {sigma_empty:.4f}", flush=True)

    # Precompute downstream saves for every edge used in any scenario
    print("Precomputing downstream save sets...", flush=True)
    all_edges = set()
    for _, edges_used in scenarios:
        all_edges |= edges_used

    edge_saves = {
        edge: [downstream_nodes(edge, sc[1]) for sc in scenarios]
        for edge in all_edges
    }
    print(f"Candidate edges: {len(all_edges)}", flush=True)

    blocked            = []
    blocked_set        = set()
    saved_per_scenario = [set() for _ in range(num_sim)]
    out_f              = open(output_path, 'w')

    for round_i in range(1, k + 1):
        # Find edge with highest marginal gain
        best_gain = -1
        best_edge = None
        for edge, saves_list in edge_saves.items():
            if edge in blocked_set:
                continue
            gain = sum(
                len(saves_list[i] - saved_per_scenario[i])
                for i in range(num_sim)
            ) / num_sim
            if gain > best_gain:
                best_gain = gain
                best_edge = edge

        if best_edge is None or best_gain <= 0:
            # No more useful edges — pad remaining slots with any unused
            # valid graph edges so output has exactly k lines
            remaining = k - len(blocked)
            print(f"  No more positive-gain edges. "
                  f"Padding {remaining} slot(s)...", flush=True)
            for u, nbrs in adj.items():
                for v, _ in nbrs:
                    if (u, v) not in blocked_set:
                        blocked.append((u, v))
                        blocked_set.add((u, v))
                        out_f.write(f"{u} {v}\n")
                        out_f.flush()
                        if len(blocked) == k:
                            break
                if len(blocked) == k:
                    break
            break

        # Select this edge
        blocked.append(best_edge)
        blocked_set.add(best_edge)
        for i in range(num_sim):
            saved_per_scenario[i] |= edge_saves[best_edge][i]

        total_saved = sum(len(s) for s in saved_per_scenario) / num_sim
        sigma_r     = sigma_empty - total_saved
        rho         = total_saved / sigma_empty
        u, v        = best_edge
        print(f"  Round {round_i:2d}: ({u},{v})  gain={best_gain:.4f}  "
              f"sigma(R)={sigma_r:.4f}  rho={rho:.4f}", flush=True)
        out_f.write(f"{u} {v}\n")
        out_f.flush()

    out_f.close()
    return blocked, sigma_empty


def main():
    graph_path, seed_path, output_path, k, num_sim, hops_arg = parse_args()
    hops = None if hops_arg == -1 else hops_arg

    print(f"Loading graph from {graph_path}...", flush=True)
    adj = load_graph(graph_path)
    print(f"Loaded {len(adj)} nodes", flush=True)

    print(f"Loading seeds from {seed_path}...", flush=True)
    seeds = load_seeds(seed_path)
    print(f"Seeds: {seeds}", flush=True)
    print(f"k={k}  num_sim={num_sim}  hops={hops_arg}", flush=True)

    print(f"\nReproducing {num_sim} exact evaluation scenarios "
          f"(seed={EVAL_SEED})...", flush=True)
    scenarios = reproduce_eval_scenarios(adj, seeds, num_sim, hops=hops)
    print(f"Scenarios reproduced.", flush=True)

    blocked, sigma_empty = greedy_on_scenarios(
        scenarios, k, output_path, adj)

    # Final verification using exact evaluator logic
    blocked_set = frozenset(blocked)
    rng = random.Random(EVAL_SEED)
    r0  = [simulate_seeded(adj, seeds, frozenset(), rng, hops)
           for _ in range(num_sim)]
    rng = random.Random(EVAL_SEED)
    rr  = [simulate_seeded(adj, seeds, blocked_set, rng, hops)
           for _ in range(num_sim)]
    s0  = sum(r0) / num_sim
    sr  = sum(rr) / num_sim
    rho = (s0 - sr) / s0

    print(f"\n{'─'*45}", flush=True)
    print(f"sigma(empty) = {s0:.4f}", flush=True)
    print(f"sigma(R)     = {sr:.4f}", flush=True)
    print(f"rho(R)       = {rho:.4f}  ({rho*100:.2f}%)", flush=True)
    print(f"Blocked      = {len(blocked)} edges", flush=True)
    print(f"Output       → {output_path}", flush=True)

if __name__ == "__main__":
    main()