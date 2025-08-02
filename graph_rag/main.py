from neo4j import GraphDatabase
import json, os
import numpy as np
import faiss
import google.generativeai as genai
from pypdf import PdfReader
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# -----------------------------------
#Neo4j Configuration
# -----------------------------------
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -----------------------------------
#Gemini Setup
# -----------------------------------
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------------
#  1. Basic DB check
# -----------------------------------
def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Connected to Neo4j!' AS message")
        for record in result:
            print(record["message"])

# -----------------------------------
# Add relationships
# -----------------------------------
def add_relation(person, relation, target):
    clean_relation = relation.upper().replace(" ", "_").replace("-", "_").replace(".", "")
    clean_relation = ''.join(c if c.isalnum() or c == '_' else '' for c in clean_relation)
    
    with driver.session() as session:
        session.run(f"""
            MERGE (p:Entity {{name: $person}})
            MERGE (t:Entity {{name: $target}})
            MERGE (p)-[:{clean_relation}]->(t)
        """, {"person": person, "target": target})
    print(f"✅ Added: {person} -[{clean_relation}]-> {target}")

# -----------------------------------
# Read PDF text
# -----------------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# -----------------------------------
# Extract relationships (LLM)
# -----------------------------------
def extract_relationships(text):
    prompt = f"""
Extract relationships from this text. ONLY return a valid JSON array.
Use simple relations: MANAGES, REPORTS_TO, SUPERVISES, ASSISTS, WORKS_UNDER.

Text:
{text}
"""
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.replace("```json", "").replace("```", "")
    return json.loads(raw_text)

# -----------------------------------
# Chunk text for FAISS
# -----------------------------------
def chunk_text(text, chunk_size=300):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# -----------------------------------
#Embeddings + FAISS
# -----------------------------------
def get_embedding(text):
    embed = genai.embed_content(model="models/embedding-001", content=text)
    return np.array(embed["embedding"])

def create_faiss_index(chunks):
    embeddings = np.array([get_embedding(chunk) for chunk in chunks]).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ Added {index.ntotal} chunks to FAISS index")
    return index, dimension

def search_vector(question, index, chunks, k=3):
    """
    Search for relevant chunks using FAISS based on the question embedding.
    """
    if isinstance(k, list):  # 💡 Ensure k is an integer
        k = k[0] if k else 3

    query_vector = get_embedding(question).reshape(1, -1)
    _, I = index.search(query_vector, k)
    
    return [chunks[i] for i in I[0] if i < len(chunks)]


# def graph_rag_answer(question, index, chunks):
#     """
#     Generate an answer by combining vector similarity and knowledge graph querying.
#     """
#     vector_context = "\n".join(search_vector(question, index, chunks, k=3))
    
#     # This part is simplified. You can expand it as needed.
#     if vector_context:
#         return f"Answer based on context:\n{vector_context}"
#     else:
#         return "Sorry, I couldn't find a relevant answer in the knowledge base."

# -----------------------------------
# Graph Query Helpers
# -----------------------------------
def query_graph(query, params=None):
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]

def debug_relationships():
    with driver.session() as session:
        result = session.run("MATCH (a)-[r]->(b) RETURN a.name, type(r) AS relation, b.name ORDER BY a.name")
        relationships = [record.data() for record in result]
        print("🐛 DEBUG: All relationships in the graph:")
        for rel in relationships:
            print(f"{rel['a.name']} -[{rel['relation']}]-> {rel['b.name']}")

# -----------------------------------
#GraphRAG answer generation
# -----------------------------------
def graph_rag_answer(question, index, chunks):
    vector_context = "\n".join(search_vector(question, index, chunks))
    
    graph_data = query_graph("MATCH (a)-[r]->(b) RETURN a.name, type(r) AS relation, b.name LIMIT 50")
    graph_text = "\n".join([f"{row['a.name']} -[{row['relation']}]-> {row['b.name']}" for row in graph_data])

    prompt = f"""
Use the following graph relationships + context to answer the question:

Context:
{vector_context}

Graph:
{graph_text}

Question: {question}
"""
    response = model.generate_content(prompt)
    return response.text

# -----------------------------------
#Main pipeline
# -----------------------------------
def main():
    # 1. Test Neo4j connection
    test_connection()

    # 2. Ingest sample (or use `read_pdf`)
    sample_text = """
Upload PDF
"""
    # 3. Extract relationships
    relations = extract_relationships(sample_text)
    for rel in relations:
        add_relation(rel["person"], rel["relation"], rel["target"])

    # 4. Build FAISS index
    global chunks
    chunks = chunk_text(sample_text)
    index, _ = create_faiss_index(chunks)

    # 5. Debug graph
    debug_relationships()

    # 6. Ask a question
    answer = graph_rag_answer("Ask Question", index, chunks)
    print("\n Answer:", answer)

if __name__ == "__main__":
    main()
