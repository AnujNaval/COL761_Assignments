import subprocess
import time
import argparse
import os
import sys

def parse_dataset_robust(filepath):
    """
    Parses various graph formats including Assignment PDF format.
    """
    graphs = []
    
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print("Error: Dataset file is empty.")
        sys.exit(1)

    is_standard = lines[0].startswith('t')
    
    if is_standard:
        current_graph = None
        for line in lines:
            parts = line.split()
            if line.startswith('t'):
                if current_graph: graphs.append(current_graph)
                gid = parts[2] if len(parts) > 2 else len(graphs)
                current_graph = {'id': gid, 'nodes': [], 'edges': []}
            elif line.startswith('v'):
                if len(parts) >= 3:
                    current_graph['nodes'].append(parts[2])
            elif line.startswith('e') or line.startswith('u'):
                if len(parts) >= 4:
                    current_graph['edges'].append((parts[1], parts[2], parts[3]))
        if current_graph: graphs.append(current_graph)

    else:
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#'):
                graph_id = line.replace('#', '').strip()
                current_graph = {'id': graph_id, 'nodes': [], 'edges': []}
                graphs.append(current_graph)
                i += 1
                
                if i >= len(lines): break
                if lines[i].startswith('#'): continue 

                try:
                    num_nodes = int(lines[i])
                    i += 1
                    for _ in range(num_nodes):
                        if i < len(lines):
                            if lines[i].startswith('#'): break 
                            current_graph['nodes'].append(lines[i])
                            i += 1
                    
                    if i < len(lines) and not lines[i].startswith('#'):
                        try:
                            num_edges = int(lines[i])
                            i += 1
                            for _ in range(num_edges):
                                if i < len(lines):
                                    if lines[i].startswith('#'): break
                                    edge_parts = lines[i].replace(',', ' ').split()
                                    if len(edge_parts) >= 3:
                                        current_graph['edges'].append(edge_parts[:3])
                                    i += 1
                        except ValueError: pass
                except ValueError: pass
            else:
                i += 1

    return graphs

def write_gspan_format_with_mapping(graphs, output_path):
    """Writes gSpan format with Integer mapping."""
    node_label_map = {}
    edge_label_map = {}
    
    def get_id(val, mapping):
        if val not in mapping:
            mapping[val] = len(mapping)
        return mapping[val]

    with open(output_path, 'w') as f:
        for idx, g in enumerate(graphs):
            if not g['nodes'] and not g['edges']: continue
            
            f.write(f"t # {idx}\n")
            for nid, label in enumerate(g['nodes']):
                f.write(f"v {nid} {get_id(label, node_label_map)}\n")
            for u, v, l in g['edges']:
                f.write(f"e {u} {v} {get_id(l, edge_label_map)}\n")

def write_fsg_format_with_mapping(graphs, output_path):
    """Writes FSG format with Integer mapping."""
    node_label_map = {}
    edge_label_map = {}
    
    def get_id(val, mapping):
        if val not in mapping:
            mapping[val] = len(mapping)
        return mapping[val]

    with open(output_path, 'w') as f:
        valid_count = 0
        for idx, g in enumerate(graphs):
            if not g['nodes'] and not g['edges']: continue
            
            f.write(f"t # {valid_count}\n")
            valid_count += 1
            
            for nid, label in enumerate(g['nodes']):
                f.write(f"v {nid} {get_id(label, node_label_map)}\n")
            
            for u, v, l in g['edges']:
                f.write(f"u {u} {v} {get_id(l, edge_label_map)}\n")

def run_mining(binary, dataset, support_pct, output_file, algorithm, total_graphs):
    abs_support = int((support_pct / 100.0) * total_graphs)
    if abs_support == 0: abs_support = 1
    
    cmd = []
    if algorithm == 'gspan':
        s_val = support_pct / 100.0
        cmd = [binary, '-f', dataset, '-s', str(s_val), '-o', output_file]
    elif algorithm == 'fsg':
        cmd = [binary, '-s', str(support_pct), dataset]
    elif algorithm == 'gaston':
        cmd = [binary, str(abs_support), dataset, output_file]

    print(f"Running {algorithm} at {support_pct}%...")
    
    start_time = time.time()
    try:
        if algorithm == 'fsg':
            with open(output_file, 'w') as outfile:
                subprocess.run(cmd, stdout=outfile, stderr=subprocess.PIPE, check=True)
        else:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm}: {e}")
        if e.stderr:
            print(f"  STDERR: {e.stderr.decode().strip()}")
        open(output_file, 'a').close() 
        return 0.0
    except Exception as e:
        print(f"Execution failed: {e}")
        return 0.0

    end_time = time.time()
    return end_time - start_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gspan', required=True)
    parser.add_argument('--fsg', required=True)
    parser.add_argument('--gaston', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    print(f"Parsing dataset: {args.dataset}")
    graphs = parse_dataset_robust(args.dataset)
    
    valid_graphs = [g for g in graphs if g['nodes'] or g['edges']]
    total_graphs = len(valid_graphs)
    print(f"Total valid graphs loaded: {total_graphs}")
    
    if total_graphs == 0:
        print("Error: No valid graphs found.")
        sys.exit(1)

    gspan_data_path = os.path.join(args.outdir, "dataset_gspan_gaston_mapped.txt")
    fsg_data_path = os.path.join(args.outdir, "dataset_fsg.txt")
    
    write_gspan_format_with_mapping(valid_graphs, gspan_data_path)
    write_fsg_format_with_mapping(valid_graphs, fsg_data_path)

    supports = [95, 50, 25, 10, 5]
    results = {sup: {'gspan': 0, 'fsg': 0, 'gaston': 0} for sup in supports}

    for sup in supports:
        print(f"\n--- SUPPORT: {sup}% ---")
        
        # Run gSpan
        out_name = os.path.join(args.outdir, f"gspan{sup}")
        t = run_mining(args.gspan, gspan_data_path, sup, out_name, 'gspan', total_graphs)
        results[sup]['gspan'] = t

        # Run FSG
        out_name = os.path.join(args.outdir, f"fsg{sup}")
        t = run_mining(args.fsg, fsg_data_path, sup, out_name, 'fsg', total_graphs)
        results[sup]['fsg'] = t

        # Run Gaston
        out_name = os.path.join(args.outdir, f"gaston{sup}")
        t = run_mining(args.gaston, gspan_data_path, sup, out_name, 'gaston', total_graphs)
        results[sup]['gaston'] = t

    # Save results
    sorted_supports = sorted(supports)
    with open(os.path.join(args.outdir, "timing_results.txt"), "w") as f:
        f.write(f"Supports: {','.join(map(str, sorted_supports))}\n")
        f.write(f"gSpan: {','.join([str(results[s]['gspan']) for s in sorted_supports])}\n")
        f.write(f"FSG: {','.join([str(results[s]['fsg']) for s in sorted_supports])}\n")
        f.write(f"Gaston: {','.join([str(results[s]['gaston']) for s in sorted_supports])}\n")

if __name__ == "__main__":
    main()