import os
import argparse
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n" + "="*60)
    print("[!] Matplotlib is not installed on this system.")
    print("[!] PLOT GENERATION SKIPPED.")
    print("[!] However, mining is complete and data is saved.")
    print("[!] Please copy 'timing_results.txt' from your output folder")
    print("[!] to your local machine and run this script there.")
    print("="*60 + "\n")
    sys.exit(0) 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()

    timing_file = os.path.join(args.outdir, "timing_results.txt")
    
    supports = []
    times = {}

    try:
        with open(timing_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Supports:"):
                    supports = list(map(int, line.split(":")[1].strip().split(',')))
                elif line.startswith("gSpan:"):
                    times['gSpan'] = list(map(float, line.split(":")[1].strip().split(',')))
                elif line.startswith("FSG:"):
                    times['FSG'] = list(map(float, line.split(":")[1].strip().split(',')))
                elif line.startswith("Gaston:"):
                    times['Gaston'] = list(map(float, line.split(":")[1].strip().split(',')))
    except FileNotFoundError:
        print(f"Timing results file not found at: {timing_file}")
        print("Please ensure runner.py finished successfully.")
        return
    
    if not supports or not times:
        print("Timing data is empty or incomplete. Cannot plot.")
        return
    
    try:
        plt.figure(figsize=(10, 6))
        
        markers = {'gSpan': 'o', 'FSG': 's', 'Gaston': '^'}
        colors = {'gSpan': 'blue', 'FSG': 'red', 'Gaston': 'green'}

        for algo, data in times.items():
            if len(data) == len(supports):
                plt.plot(supports, data, marker=markers.get(algo, 'x'), 
                         label=algo, color=colors.get(algo, 'black'), linewidth=2)
            else:
                print(f"Warning: Data length mismatch for {algo}. Skipping.")

        plt.xlabel('Support Threshold (%)', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.title('Performance Comparison: gSpan vs FSG vs Gaston', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(supports)
        
        output_path = os.path.join(args.outdir, 'plot.png')
        plt.savefig(output_path)
        print(f"Plot successfully saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    main()