# src/search.py
# Final Clean Version
# Mistral LLM + CLIP + YOLO + FAISS + Auto Save JSON/CSV

import os
import json
import csv
import re
import faiss
import numpy as np
import torch
import ollama
from transformers import CLIPProcessor, CLIPModel


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL = "mistral"

INDEX_PATH = "index/faiss.index"
FRAME_PATHS = "embeddings/frame_paths.npy"
YOLO_META = "results/metadata/frame_objects.json"

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# LOAD CLIP
# --------------------------------------------------
print("Using device:", device)

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def format_timestamp(seconds):
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


def load_metadata():

    if not os.path.exists(YOLO_META):
        return {}

    with open(YOLO_META, "r") as f:
        data = json.load(f)

    meta = {}

    for item in data:
        meta[item["frame"]] = item

    return meta


# --------------------------------------------------
# MISTRAL QUERY PARSER
# --------------------------------------------------
def understand_query(query):

    prompt = f"""
You are an AI planner for a video search engine.

Return ONLY valid JSON.

Query:
{query}

Schema:
{{
 "objects": [],
 "colors": [],
 "actions": [],
 "relations": [],
 "time_filter": ""
}}

Examples:

white car ->
{{"objects":["car"],"colors":["white"],"actions":[],"relations":[],"time_filter":""}}

man near bus ->
{{"objects":["person","bus"],"colors":[],"actions":[],"relations":["near"],"time_filter":""}}
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response["message"]["content"].strip()

        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match:
            json_text = match.group(0)
            plan = json.loads(json_text)
        else:
            raise Exception("No JSON found")

    except:

        q = query.lower()

        plan = {
            "objects": [],
            "colors": [],
            "actions": [],
            "relations": [],
            "time_filter": ""
        }

        if "car" in q:
            plan["objects"].append("car")

        if "bus" in q:
            plan["objects"].append("bus")

        if any(word in q for word in [
            "man", "woman", "boy",
            "girl", "person", "people"
        ]):
            plan["objects"].append("person")

        for color in [
            "white", "black", "red",
            "blue", "green", "yellow"
        ]:
            if color in q:
                plan["colors"].append(color)

    return plan


# --------------------------------------------------
# SEARCH
# --------------------------------------------------
def search(query, k=5):

    if not query.strip():
        return []

    if not os.path.exists(INDEX_PATH):
        return []

    if not os.path.exists(FRAME_PATHS):
        return []

    try:
        index = faiss.read_index(INDEX_PATH)
        frame_paths = np.load(FRAME_PATHS, allow_pickle=True)
        metadata = load_metadata()

        # Query Plan
        plan = understand_query(query)

        requested_objects = plan["objects"]
        requested_colors = plan["colors"]

        # ------------------------------------------
        # CLIP TEXT EMBEDDING
        # ------------------------------------------
        with torch.no_grad():

            inputs = processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }

            text_features = model.get_text_features(**inputs)

            if hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output

            norm = text_features.norm(
                p=2,
                dim=-1,
                keepdim=True
            )

            text_features = text_features / norm.clamp(min=1e-12)

            query_embedding = text_features.cpu().numpy().astype("float32")

        # ------------------------------------------
        # FAISS SEARCH
        # ------------------------------------------
        distances, indices = index.search(
            query_embedding,
            max(k * 20, 100)
        )

        results = []

        for idx, dist in zip(indices[0], distances[0]):

            if idx >= len(frame_paths):
                continue

            frame_path = frame_paths[idx]

            filename = os.path.basename(frame_path)

            try:
                seconds = int(
                    filename.split("_")[-1].replace(".jpg", "")
                )
            except:
                seconds = 0

            meta = metadata.get(frame_path, {
                "objects": []
            })

            detected_objects = meta.get("objects", [])

            boost = 0
            penalty = 0

            # Object Matching
            for obj in requested_objects:

                if obj in detected_objects:
                    boost += 0.45
                else:
                    penalty += 0.20

            # Color Intent
            if requested_colors:
                boost += 0.10

            final_score = float(dist) - boost + penalty

            results.append({
                "frame": frame_path,
                "timestamp": format_timestamp(seconds),
                "score": round(final_score, 4),
                "objects": detected_objects,
                "query_plan": plan
            })

        results = sorted(
            results,
            key=lambda x: x["score"]
        )

        return results[:k]

    except Exception as e:
        print("Search Error:", e)
        return []


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
def save_results(query, results):

    os.makedirs("results", exist_ok=True)

    # JSON
    json_data = []

    for r in results:
        json_data.append({
            "query": query,
            "timestamp": r["timestamp"],
            "frame": r["frame"],
            "score": r["score"],
            "objects": r["objects"]
        })

    with open("results/results.json", "w") as f:
        json.dump(json_data, f, indent=4)

    # CSV
    with open(
        "results/results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "query",
            "timestamp",
            "frame",
            "score",
            "objects"
        ])

        for r in results:
            writer.writerow([
                query,
                r["timestamp"],
                r["frame"],
                r["score"],
                ", ".join(r["objects"])
            ])


# --------------------------------------------------
# RUN DIRECTLY
# --------------------------------------------------
if __name__ == "__main__":

    while True:

        q = input("\nEnter query: ").strip()

        if q.lower() == "exit":
            break

        output = search(q)

        save_results(q, output)

        for i, item in enumerate(output, start=1):

            print(f"\n{i}. {item['timestamp']}")
            print("Score   :", item["score"])
            print("Objects :", item["objects"])
            print("Plan    :", item["query_plan"])

        print("\nResults saved to:")
        print("results/results.json")
        print("results/results.csv")