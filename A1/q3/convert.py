import sys
import os
import networkx as nx
import numpy as np
import networkx.algorithms.isomorphism as iso

LABEL_MAPPING = {
    'Br': 0, 'C': 1, 'Cl': 2, 'F': 3, 'H': 4, 
    'I': 5, 'N': 6, 'O': 7, 'P': 8, 'S': 9, 'Si': 10
}

def parse_graph_file(file_path, is_gspan_format=False):
    graphs = []
    current_graph = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                parts = line.split()
                
                if line.startswith('#') or line.startswith('t #') or line.startswith('Graph'):
                    if current_graph is not None:
                        graphs.append(current_graph)
                    current_graph = nx.Graph()
                    continue
                
                if current_graph is None: current_graph = nx.Graph()

                if parts[0] == 'v':
                    node_id = int(parts[1])
                    raw_label = parts[2]
                    
                    if is_gspan_format:
                        label_int = int(raw_label)
                    else:
                        if raw_label.isdigit():
                            label_int = int(raw_label)
                        else:
                            label_int = LABEL_MAPPING.get(raw_label, -1)
                            
                    if label_int != -1:
                        current_graph.add_node(node_id, label=label_int)

                elif parts[0] == 'e':
                    src = int(parts[1])
                    dst = int(parts[2])
                    edge_label = int(parts[3])
                    current_graph.add_edge(src, dst, label=edge_label)
            
            if current_graph is not None:
                graphs.append(current_graph)
                
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        sys.exit(1)
        
    return graphs

def generate_features(dataset_graphs, subgraphs):
    num_graphs = len(dataset_graphs)
    num_subgraphs = len(subgraphs)
    
    print(f"Generating features for {num_graphs} graphs against {num_subgraphs} subgraphs")
    
    feature_matrix = np.zeros((num_graphs, num_subgraphs), dtype=int)
    nm = iso.categorical_node_match('label', -1)
    em = iso.categorical_edge_match('label', -1)
    
    for i, graph in enumerate(dataset_graphs):
        if i % 100 == 0:
            print(f"Processing graph {i}/{num_graphs}...", end='\r')
            
        for j, subgraph in enumerate(subgraphs):
            matcher = iso.GraphMatcher(graph, subgraph, node_match=nm, edge_match=em)
            if matcher.subgraph_is_isomorphic():
                feature_matrix[i, j] = 1
                
    return feature_matrix

def main():
    if len(sys.argv) != 4:
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    subgraphs_path = sys.argv[2]
    output_path = sys.argv[3]
    
    dataset_graphs = parse_graph_file(dataset_path, is_gspan_format=False)
    
    discriminative_subgraphs = parse_graph_file(subgraphs_path, is_gspan_format=True)
    
    if not discriminative_subgraphs:
        print("Error: No subgraphs loaded. Check discriminative_subgraphs.txt")
        sys.exit(1)
        
    features = generate_features(dataset_graphs, discriminative_subgraphs)
    
    np.save(output_path, features)

if __name__ == "__main__":
    main()