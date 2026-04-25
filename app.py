# app.py

import os
import streamlit as st
from PIL import Image
from src.search import search


# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Intelligent Video Search Engine",
    layout="wide"
)


# -----------------------------------
# Session State
# -----------------------------------
if "query_history" not in st.session_state:
    st.session_state.query_history = []


# -----------------------------------
# Title
# -----------------------------------
st.title("🎥 Intelligent Video Search Engine")
st.write("Search relevant video moments using natural language.")


# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Search Settings")

top_k = st.sidebar.slider(
    "Number of Results",
    min_value=1,
    max_value=10,
    value=5
)

if st.sidebar.button("Clear History"):
    st.session_state.query_history = []


# -----------------------------------
# Query Form
# -----------------------------------
with st.form("search_form", clear_on_submit=True):
    query = st.text_input("Enter your query")
    submit = st.form_submit_button("Search")


# -----------------------------------
# Run Search
# -----------------------------------
if submit and query.strip():

    with st.spinner("Searching..."):
        results = search(query.strip(), k=top_k)

    st.session_state.query_history.insert(0, {
        "query": query.strip(),
        "results": results
    })


# -----------------------------------
# Display Query History
# -----------------------------------
for item in st.session_state.query_history:

    st.markdown("---")
    st.header(f"Query: {item['query']}")

    results = item["results"]

    if not results:
        st.warning("No results found.")
        continue

    st.success(f"Found {len(results)} results")

    for i, r in enumerate(results, start=1):

        st.subheader(f"Result {i}")

        col1, col2 = st.columns([2, 1])

        # -----------------------------------
        # Show Bounding Box Image
        # -----------------------------------
        with col1:

            annotated_path = r["frame"].replace(
                "data/frames",
                "results/annotated_frames"
            )

            if os.path.exists(annotated_path):
                img = Image.open(annotated_path)
                st.image(img, use_container_width=True)

            elif os.path.exists(r["frame"]):
                img = Image.open(r["frame"])
                st.image(img, use_container_width=True)

            else:
                st.error("Image not found.")

        # -----------------------------------
        # Metadata
        # -----------------------------------
        with col2:
            st.write(f"**Timestamp:** {r['timestamp']}")
            st.write(f"**Score:** {r['score']}")
            st.write(f"**Frame Path:** {r['frame']}")

            if "objects" in r and r["objects"]:
                st.write("**Detected Objects:**")
                for obj in r["objects"]:
                    st.success(obj)
            else:
                st.write("No detected objects")


# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption("Built with CLIP + YOLOv8 + FAISS + Streamlit")