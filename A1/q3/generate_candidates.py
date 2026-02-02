import sys
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 generate_candidates.py <db_features_path> <query_features_path> <output_path>")
        sys.exit(1)

    db_features_path = sys.argv[1]
    query_features_path = sys.argv[2]
    output_path = sys.argv[3]

    print("Loading feature matrices...")
    try:
        db_feats = np.load(db_features_path)
        query_feats = np.load(query_features_path)
    except Exception as e:
        print(f"Error loading numpy files: {e}")
        sys.exit(1)

    num_queries = query_feats.shape[0]
    num_db = db_feats.shape[0]
    
    print(f"Processing {num_queries} queries against {num_db} database graphs...")

    try:
        with open(output_path, 'w') as f:
            for i in range(num_queries):
                q_vec = query_feats[i]
                
                comparison = (q_vec <= db_feats)
                candidates_mask = np.all(comparison, axis=1)
                
                candidate_indices = np.where(candidates_mask)[0]
                
                candidates_str = " ".join(map(str, candidate_indices))
                
                f.write(f"q # {i}\n")
                f.write(f"c # {candidates_str}\n")
                
                if i % 100 == 0:
                    print(f"Processed {i}/{num_queries} queries...", end='\r')

    except IOError as e:
        print(f"Error writing to output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()