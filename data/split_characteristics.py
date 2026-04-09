import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def split_data(input_file, output_train, output_val, ratio=0.3):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    n_samples = len(lines)
    target_30 = int(n_samples * ratio)
    target_70 = n_samples - target_30
    
    print(f"Total lines: {n_samples}. Goal: {target_30} / {target_70} split.")

    # 1. Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(lines)

    # 2. Apply K-Means with K=2 to find two distinct poles
    # We use a fixed random state for reproducibility
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # 3. Get distances to centroids
    # kmeans.transform() returns distance to each cluster center
    distances = kmeans.transform(X)
    
    # We want to pick the 30 samples that are collectively "most distinct" 
    # from the rest. We can do this by picking the 30 samples closest to 
    # one of the centroids (the one that naturally captures ~30% - ~50% of the data)
    # OR more simply: Rank samples by (dist_to_c0 - dist_to_c1)
    # Samples with very negative values are "very c0", very positive are "very c1"
    
    diff = distances[:, 0] - distances[:, 1]
    sorted_indices = np.argsort(diff)
    
    # Take the first 30 (most cluster 0) and the rest (most cluster 1)
    # This naturally maximizes the distance between the two sets because 
    # we are splitting at the most discriminative boundary.
    set_30_indices = sorted_indices[:target_30]
    set_70_indices = sorted_indices[target_30:]
    
    lines_30 = [lines[i] for i in set_30_indices]
    lines_70 = [lines[i] for i in set_70_indices]
    
    print(f"Writing {len(lines_30)} lines to {output_val}...")
    with open(output_val, 'w') as f:
        f.write('\n'.join(lines_30) + '\n')
        
    print(f"Writing {len(lines_70)} lines to {output_train}...")
    with open(output_train, 'w') as f:
        f.write('\n'.join(lines_70) + '\n')

    # 4. Visualization
    print("Generating visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 7))
    
    # Plot set_70 (train)
    plt.scatter(X_2d[set_70_indices, 0], X_2d[set_70_indices, 1], 
                c='blue', label='Train (70%)', alpha=0.6)
    
    # Plot set_30 (val)
    plt.scatter(X_2d[set_30_indices, 0], X_2d[set_30_indices, 1], 
                c='red', label='Val (30%)', alpha=0.6)
    
    plt.title("2D Projection of Data Split (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(os.path.dirname(input_file), "split_visualization.png")
    plt.savefig(plot_path)
    print(f"Visualization saved to {plot_path}")

    print("Done!")

if __name__ == "__main__":
    input_path = "./toy/toy_tonality.txt"
    output_train_path = "/home/toni/Documents/PROJECTS/llm_concept_learning/benchmarks/toy/binary_answer/characteristics_train.txt"
    output_val_path = "/home/toni/Documents/PROJECTS/llm_concept_learning/benchmarks/toy/binary_answer/characteristics_val.txt"
    
    split_data(input_path, output_train_path, output_val_path)
