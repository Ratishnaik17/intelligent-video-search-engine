# main.py
# Final Submission Version

from src.extract_frames import process_all_videos
from src.detect import detect_objects
from src.embed import generate_embeddings
from src.index import build_index
from src.search import search


def main():

    print("=" * 55)
    print("🎥 Intelligent Video Search Engine")
    print("=" * 55)

    # -----------------------------------
    # Step 1: Extract Frames
    # -----------------------------------
    print("\n[1/5] Extracting frames from videos...")
    process_all_videos(fps=1)

    # -----------------------------------
    # Step 2: Detect Objects using YOLOv8
    # -----------------------------------
    print("\n[2/5] Detecting objects in frames...")
    detect_objects()

    # -----------------------------------
    # Step 3: Generate CLIP Embeddings
    # -----------------------------------
    print("\n[3/5] Generating image embeddings...")
    generate_embeddings()

    # -----------------------------------
    # Step 4: Build FAISS Index
    # -----------------------------------
    print("\n[4/5] Building FAISS index...")
    build_index()

    # -----------------------------------
    # Step 5: Search Loop
    # -----------------------------------
    print("\n[5/5] Search Engine Ready!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("Enter search query: ").strip()

        if query.lower() == "exit":
            print("Exiting search engine.")
            break

        results = search(query, k=5)

        if not results:
            print("No results found.\n")
            continue

        print("\nTop Results:\n")

        for i, result in enumerate(results, start=1):

            print(f"{i}. Timestamp : {result['timestamp']}")
            print(f"   Score     : {result['score']}")
            print(f"   Objects   : {result['objects']}")
            print(f"   Frame     : {result['frame']}")
            print("-" * 45)


if __name__ == "__main__":
    main()