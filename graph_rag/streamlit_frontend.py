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

st.set_page_config(page_title="GraphRAG Chat", layout="wide")
st.title("🧠 GraphRAG with Neo4j, FAISS & Gemini")
st.markdown("Ask natural language questions, powered by graph + semantic search.")

# Session state init
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.data_loaded = False

# --- Upload and Text Input ---
st.markdown("## 📄 Upload a File or Paste Text")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
# sample_text = st.text_area(
#     "Or paste your text here",
#     height=200,
#     placeholder="Enter text to extract relationships and build the graph..."
# )

if st.button("🔄 Process Input"):
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            sample_text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(uploaded_file)
            sample_text = " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

    if not sample_text.strip():
        st.error("❗ Please upload a file or paste some text.")
    else:
        st.write("🔗 Connecting to Neo4j...")
        test_connection()

        st.write("🔍 Extracting relationships...")
        relationships = extract_relationships(sample_text)

        st.write("🧩 Adding to graph...")
        for rel in relationships:
            add_relation(rel["person"], rel["relation"], rel["target"])

        st.write("✂️ Chunking and indexing text...")
        chunks = chunk_text(sample_text)
        index, _ = create_faiss_index(chunks)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.session_state.data_loaded = True

        st.success("✅ Data loaded successfully!")

# --- Main Q&A Interface ---
if st.session_state.data_loaded:
    st.markdown("### 💬 Ask a Question")
    question = st.text_input("Enter your question", value="Enter your question here")
    if st.button("Ask"):
        with st.spinner("Generating answer..."):
            answer = graph_rag_answer(question, st.session_state.index, st.session_state.chunks)
            st.success("🤖 Gemini's Answer")
            st.markdown(f"**{answer.strip()}**")
else:
    st.warning("⬆️ Please upload a file or paste text above to start.")
