import streamlit as st
from main import (
    test_connection,
    extract_relationships,
    add_relation,
    chunk_text,
    create_faiss_index,
    debug_relationships,
    graph_rag_answer,
)

# Initialize Streamlit UI
st.set_page_config(page_title="GraphRAG Chat", layout="wide")
st.title("🧠 GraphRAG with Neo4j, FAISS & Gemini")
st.markdown("Ask natural language questions, powered by graph + semantic search.")

# Session state init
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.data_loaded = False

# --- Sidebar Setup ---
st.sidebar.header("📄 Upload & Setup")
sample_text = st.sidebar.text_area("Paste your sample input text", height=200, value="Upload PDF or paste text here to extract relationships and build the graph.")

if st.sidebar.button("🔄 SEND"):
    st.sidebar.write("Connecting to Neo4j...")
    test_connection()

    st.sidebar.write("🔍 Extracting relationships...")
    relationships = extract_relationships(sample_text)

    st.sidebar.write("🧩 Adding to graph...")
    for rel in relationships:
        add_relation(rel["person"], rel["relation"], rel["target"])

    st.sidebar.write("✂️ Chunking and indexing text...")
    chunks = chunk_text(sample_text)
    index, _ = create_faiss_index(chunks)

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.data_loaded = True

    st.success("✅ Data loaded successfully!")

# --- Main Interface ---
if st.session_state.data_loaded:
    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Enter your question", value="Enter your question")
    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            answer = graph_rag_answer(question, st.session_state.index, st.session_state.chunks)
            st.success("🤖 Gemini's Answer")
            st.markdown(f"**{answer.strip()}**")

    # with st.expander("🛠️ Debug: Show Graph Relationships"):
    #     debug_relationships()
else:
    st.warning("⬅️ Upload Your PDF and Ask Questions.")
