# src/embed.py

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# Increase timeout for HuggingFace Hub downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

# -----------------------------
# Configuration
# -----------------------------
FRAME_FOLDER = "data/frames"
SAVE_EMBEDDINGS = "embeddings/embeddings.npy"
SAVE_PATHS = "embeddings/frame_paths.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"

# -----------------------------
# Device Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load CLIP Model
# -----------------------------
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


# -----------------------------
# Generate Embeddings
# -----------------------------
def generate_embeddings():
    os.makedirs("embeddings", exist_ok=True)

    frame_files = sorted(
        [f for f in os.listdir(FRAME_FOLDER)
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )

    if not frame_files:
        print("No frames found in data/frames")
        return

    embeddings = []
    frame_paths = []

    print(f"Generating embeddings for {len(frame_files)} frames...\n")

    with torch.no_grad():
        for file in tqdm(frame_files):
            path = os.path.join(FRAME_FOLDER, file)

            image = Image.open(path).convert("RGB")

            inputs = processor(images=image, return_tensors="pt").to(device)

            image_features = model.get_image_features(**inputs).pooler_output

            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings.append(image_features.cpu().numpy()[0])
            frame_paths.append(path)

    embeddings = np.array(embeddings).astype("float32")

    np.save(SAVE_EMBEDDINGS, embeddings)
    np.save(SAVE_PATHS, np.array(frame_paths))

    print("\nEmbeddings saved successfully!")
    print(f"Saved: {SAVE_EMBEDDINGS}")
    print(f"Saved: {SAVE_PATHS}")
    print(f"Shape: {embeddings.shape}")


# -----------------------------
# Run File Directly
# -----------------------------
if __name__ == "__main__":
    generate_embeddings()