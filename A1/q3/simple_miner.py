import sys
import networkx as nx
import collections
import itertools

LABEL_MAPPING = {
    'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 
    'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
}
REVERSE_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def load_graphs(file_path):
    graphs = []
    current_graph = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            
            if line.startswith('#') or line.startswith('t #') or line.startswith('Graph'):
                if current_graph: graphs.append(current_graph)
                current_graph = nx.Graph()
                continue
            
            if current_graph is None: current_graph = nx.Graph()
            
            if parts[0] == 'v':
                nid, lbl = int(parts[1]), parts[2]
                # Hybrid conversion
                if lbl.isdigit(): lbl = int(lbl)
                else: lbl = LABEL_MAPPING.get(lbl, -1)
                current_graph.add_node(nid, label=lbl)
                
            elif parts[0] == 'e':
                u, v, lbl = int(parts[1]), int(parts[2]), int(parts[3])
                current_graph.add_edge(u, v, label=lbl)
                
    if current_graph: graphs.append(current_graph)
    return graphs

def get_canon_edge(u_lbl, v_lbl, e_lbl):
    if u_lbl > v_lbl: return (v_lbl, u_lbl, e_lbl)
    return (u_lbl, v_lbl, e_lbl)

def mine_features(graphs, top_k=50):
    print(f"Mining features from {len(graphs)} graphs (Python Fallback)...")
    
    edge_counts = collections.Counter()
    for G in graphs:
        seen_edges = set()
        for u, v, data in G.edges(data=True):
            elbl = data.get('label', 0)
            ulbl = G.nodes[u].get('label', 0)
            vlbl = G.nodes[v].get('label', 0)
            
            canon = get_canon_edge(ulbl, vlbl, elbl)
            if canon not in seen_edges:
                edge_counts[canon] += 1
                seen_edges.add(canon)
                
    top_patterns = edge_counts.most_common(top_k)
    
    mined_subgraphs = []
    for (u_lbl, v_lbl, e_lbl), count in top_patterns:
        Sub = nx.Graph()
        Sub.add_node(0, label=u_lbl)
        Sub.add_node(1, label=v_lbl)
        Sub.add_edge(0, 1, label=e_lbl)
        mined_subgraphs.append(Sub)
        
    return mined_subgraphs

def save_subgraphs(subgraphs, output_path):
    with open(output_path, 'w') as f:
        for idx, G in enumerate(subgraphs):
            f.write(f"t # {idx}\n")
            # Write nodes
            for n, data in G.nodes(data=True):
                f.write(f"v {n} {data['label']}\n")
            # Write edges
            for u, v, data in G.edges(data=True):
                f.write(f"e {u} {v} {data['label']}\n")
    print(f"Saved {len(subgraphs)} features to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 simple_miner.py <input> <output>")
        sys.exit(1)
        
    graphs = load_graphs(sys.argv[1])
    features = mine_features(graphs, top_k=50)
    save_subgraphs(features, sys.argv[2])