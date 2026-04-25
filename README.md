#  Intelligent Video Search Engine

Demo Video Link=https://drive.google.com/file/d/1Fuc8hf3FK15_cPOWNToqHu4phvOsbOoU/view?usp=sharing
GitHub Repository= https://github.com/Ratishnaik17/intelligent-video-search-engine

An AI-powered video retrieval system that allows users to search relevant moments inside videos using **natural language queries** such as:

- `white car`
- `person near bus`
- `man walking`
- `car in parking area`

The system combines **Computer Vision + Vector Search + LLM Query Understanding**.

---

#  Features

 Natural Language Video Search  
 Frame Extraction from Videos  
 CLIP Image Embeddings  
 FAISS Fast Similarity Search  
 YOLOv8 Object Detection  
 Mistral LLM Query Understanding  
 Streamlit Web Interface  
 Bounding Box Visualization  
 Multi-query Search History

---

#  Tech Stack

| Technology | Purpose |
|----------|---------|
| Python | Core Programming |
| OpenCV | Video Frame Extraction |
| CLIP | Image + Text Embeddings |
| FAISS | Vector Similarity Search |
| YOLOv8 | Object Detection |
| Mistral (Ollama) | Query Understanding |
| Streamlit | Web UI |

---

#  System Architecture

```text


                ┌────────────────────────────┐
                │        Input Videos        │
                │          .mp4
                └────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Frame Extraction Module      │
              │ extract_frames.py            │
              │ Tool: OpenCV                │
              └────────────┬─────────────────┘
                           │
                           ▼
                 Saved Frames (JPEG Images)
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
┌──────────────────────┐            ┌──────────────────────┐
│ Object Detection     │            │ Image Embedding      │
│ detect.py            │            │ embed.py             │
│ Tool: YOLOv8         │            │ Tool: CLIP           │
└──────────┬───────────┘            └──────────┬───────────┘
           │                                   │
           ▼                                   ▼
Frame Metadata JSON                    Embedding Vectors
(objects, counts, boxes)              (512-D vectors)
           │                                   │
           └──────────────┬────────────────────┘
                          ▼
              ┌─────────────────────────────┐
              │ Indexing Module             │
              │ index.py                    │
              │ Tool: FAISS                │
              └────────────┬────────────────┘
                           │
                           ▼
                    Searchable Vector DB
                           │
                           ▼
                ┌────────────────────────────┐
                │ User Natural Language Query│
                │ "white car near person"    │
                └────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │ Query Understanding Layer   │
               │ search.py                   │
               │ Tool: Mistral (Ollama)     │
               └────────────┬────────────────┘
                            │
                            ▼
           Structured Query Intent (objects/colors)
                            │
                            ▼
               ┌─────────────────────────────┐
               │ CLIP Text Embedding         │
               │ search.py                   │
               └────────────┬────────────────┘
                            │
                            ▼
                FAISS Similarity Search
                            │
                            ▼
                YOLO Metadata Re-ranking
                            │
                            ▼
              Final Relevant Frames Returned
                            │
                            ▼
               ┌────────────────────────────┐
               │ Streamlit UI (app.py)      │
               │ Shows image + boxes + info │
               └────────────────────────────┘