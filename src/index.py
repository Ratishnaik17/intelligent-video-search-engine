# src/index.py

import os
import faiss
import numpy as np


def build_index(
    embedding_file="embeddings/embeddings.npy",
    index_path="index/faiss.index"
):
    """
    Build FAISS index from saved embeddings.

    Args:
        embedding_file (str): Path to .npy embeddings file
        index_path (str): Path to save FAISS index
    """

    # Create index folder if missing
    os.makedirs("index", exist_ok=True)

    # Check embeddings file exists
    if not os.path.exists(embedding_file):
        print("Embeddings file not found.")
        return

    # Load embeddings
    embeddings = np.load(embedding_file).astype("float32")

    if len(embeddings) == 0:
        print("No embeddings found.")
        return

    # Vector dimension
    dimension = embeddings.shape[1]

    print(f"Loaded {len(embeddings)} embeddings")
    print(f"Vector dimension: {dimension}")

    # Build FAISS Index
    index = faiss.IndexFlatL2(dimension)

    # Add vectors
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    print("\nFAISS index built successfully!")
    print(f"Saved at: {index_path}")
    print(f"Total vectors indexed: {index.ntotal}")


# Run directly
if __name__ == "__main__":
    build_index()