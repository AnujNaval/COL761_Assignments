import sys
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load_from_api(dataset_num):
    student_id = "cs5210130"
    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id={student_id}&dataset_num={dataset_num}"
    
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode('utf-8')
        parsed_data = json.loads(raw_data)
        return np.array(parsed_data["X"])

def get_inertias(data):
    k_values = list(range(1, 16))
    inertias = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return k_values, inertias

def find_optimal_k(k_values, inertias):
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])
    
    max_dist = -1
    optimal_k = 1
    
    for i in range(len(k_values)):
        p0 = np.array([k_values[i], inertias[i]])
        dist = np.abs(np.cross(p2 - p1, p1 - p0)) / np.linalg.norm(p2 - p1)
        if dist > max_dist:
            max_dist = dist
            optimal_k = k_values[i]
            
    return optimal_k

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 Q1.py <dataset_num> OR python3 Q1.py <path_to_dataset>.npy")
        sys.exit(1)
        
    arg = sys.argv[1]
    
    if arg.isdigit():
        data1 = load_from_api("1")
        data2 = load_from_api("2")
        
        k_vals, inertias1 = get_inertias(data1)
        _, inertias2 = get_inertias(data2)
        
        opt_k1 = find_optimal_k(k_vals, inertias1)
        opt_k2 = find_optimal_k(k_vals, inertias2)
        
        print(f"Optimal k for Dataset 1: {opt_k1}")
        print(f"Optimal k for Dataset 2: {opt_k2}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(k_vals, inertias1, marker='o', color='b')
        axes[0].set_title("Mode 1 (API) - Dataset 1")
        axes[0].set_xlabel("Number of clusters (k)")
        axes[0].set_ylabel("Objective Value")
        axes[0].grid(True)
        
        axes[1].plot(k_vals, inertias2, marker='s', color='r')
        axes[1].set_title("Mode 1 (API) - Dataset 2")
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plot.png')
        
    elif arg.endswith('.npy'):
        npy_files = [a for a in sys.argv[1:] if a.endswith('.npy')]
        
        if len(npy_files) == 1:
            data = np.load(npy_files[0])
            k_vals, inertias = get_inertias(data)
            opt_k = find_optimal_k(k_vals, inertias)
            
            print(f"Optimal k for {npy_files[0]}: {opt_k}")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(k_vals, inertias, marker='o', color='g')
            ax.set_title(f"Mode 2 (Local) - {npy_files[0]}")
            ax.set_xlabel("Number of clusters (k)")
            ax.set_ylabel("Objective Value")
            ax.grid(True)
            
            plt.savefig('plot.png')
            
        elif len(npy_files) == 2:
            data1 = np.load(npy_files[0])
            data2 = np.load(npy_files[1])
            
            k_vals, inertias1 = get_inertias(data1)
            _, inertias2 = get_inertias(data2)
            
            opt_k1 = find_optimal_k(k_vals, inertias1)
            opt_k2 = find_optimal_k(k_vals, inertias2)
            
            print(f"Optimal k for {npy_files[0]}: {opt_k1}")
            print(f"Optimal k for {npy_files[1]}: {opt_k2}")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].plot(k_vals, inertias1, marker='o', color='b')
            axes[0].set_title(f"Mode 2 (Local) - {npy_files[0]}")
            axes[0].set_xlabel("Number of clusters (k)")
            axes[0].set_ylabel("Objective Value")
            axes[0].grid(True)
            
            axes[1].plot(k_vals, inertias2, marker='s', color='r')
            axes[1].set_title(f"Mode 2 (Local) - {npy_files[1]}")
            axes[1].set_xlabel("Number of clusters (k)")
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('plot.png')

if __name__ == "__main__":
    main()