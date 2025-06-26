import streamlit as st
import tempfile
import time
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from llama_cloud_services import LlamaParse
from pdf2image import convert_from_path

from langchain.retrievers import ContextualCompressionRetriever
from sentence_transformers import CrossEncoder

# --- Page Config ---
st.set_page_config(page_title="ChatBot", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
        .stApp { background-color: black; color: white; }
        .user-bubble {
            background-color: #2a2a2a; color: white;
            padding: 10px; border-radius: 12px; max-width: 60%;
            float: right; margin: 5px 0 10px auto; text-align: right;
        }
        .assistant-bubble {
            background-color: #1e1e1e; color: white;
            padding: 10px; border-radius: 12px; max-width: 60%;
            float: left; margin: 5px auto 10px 0; text-align: left;
        }
        .clearfix::after { content: ""; display: block; clear: both; }
        .floating-button {
            position: fixed; bottom: 20px; right: 20px; z-index: 9999;
        }
        .floating-button button {
            background-color: #444 !important; color: white !important;
            padding: 10px 18px; border-radius: 999px; border: none;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Underwriting Agent")

# --- PDF Upload Logic ---
uploaded_pdf = st.file_uploader("Upload a Valuation PDF", type=["pdf"])
if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        PDF_PATH = tmp_file.name
else:
    st.stop()

# --- Session State Setup ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."})
    st.session_state.messages.append({"role": "assistant", "content": "What can I help you with?"})

valuation_clicked = st.empty()
valuation_triggered = valuation_clicked.button("Valuation 💰", key="valuation_btn", help="Click to ask about valuation")
user_input = st.chat_input("Message...")
prompt = "What is the valuation?" if valuation_triggered else user_input

# --- Query Preprocessing ---
def preprocess_query(query):
    q = query.lower()
    if "adjusted" in q and "ebitda" in q:
        return query + " (not just EBITDA, focus on adjusted EBITDA only)"
    if "net sales" in q and any(y in q for y in ["2021", "2022", "2023", "2024", "2025"]):
        return query + " (make sure year is exactly correct)"
    return query

# --- Parse PDF ---
def parse_pdf():
    parser = LlamaParse(
        api_key=st.secrets["LLAMA_CLOUD_API_KEY"],
        num_workers=4,
        verbose=False,
        language="en"
    )
    result = parser.parse(PDF_PATH)
    lc_documents = []
    for page in result.pages:
        if page.md.strip():
            content = page.md.strip()
            is_table = ('|' in content and any(line.strip().startswith('|') and '---' in line for line in content.splitlines()))
            metadata = {"page_number": page.page}
            if is_table:
                metadata["type"] = "table"
            lc_documents.append(Document(page_content=content, metadata=metadata))
    return lc_documents

# --- FinBERT Embedding Setup ---
def get_vectorstores(docs):
    embed = HuggingFaceEmbeddings(model_name="yiyanghkust/finbert-tone")
    table_docs = [doc for doc in docs if doc.metadata.get("type") == "table"]
    full_vs = FAISS.from_documents(docs, embed)
    table_vs = FAISS.from_documents(table_docs, embed)
    return full_vs.as_retriever(), table_vs.as_retriever()

# --- HuggingFace Manual Reranker ---
class HuggingFaceReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def compress_documents(self, documents, query):
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        sorted_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:3]]  # top 3 only

# --- QA Chain Setup with HF Reranker ---
def get_qa_chains(full_ret, table_ret):
    llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

    reranker = HuggingFaceReranker()
    reranked_full_ret = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=full_ret)
    reranked_table_ret = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=table_ret)

    custom_prompt = PromptTemplate.from_template("""
You are a helpful financial assistant. Based on the context below, extract markdown tables exactly as is without any modifications or reformatting. Do not summarize. Only include the table using proper markdown format.

Context:
{context}

Question:
{question}
""")

    qa_full = RetrievalQA.from_chain_type(llm=llm, retriever=reranked_full_ret, return_source_documents=True)
    qa_table = RetrievalQA.from_chain_type(llm=llm, retriever=reranked_table_ret, return_source_documents=True, chain_type_kwargs={"prompt": custom_prompt})
    return qa_full, qa_table

# --- First Run Initialization ---
if not st.session_state.initialized:
    with st.spinner("Parsing PDF..."):
        docs = parse_pdf()
    with st.spinner("Building vectorstore..."):
        full_ret, table_ret = get_vectorstores(docs)
    with st.spinner("Setting up QA chains..."):
        qa_full, qa_table = get_qa_chains(full_ret, table_ret)
    st.session_state.docs = docs
    st.session_state.full_retriever = full_ret
    st.session_state.table_retriever = table_ret
    st.session_state.qa_chain_full = qa_full
    st.session_state.qa_chain_table = qa_table
    st.session_state.initialized = True

# --- Show Chat History ---
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

# --- Typing Animation ---
def typewriter_output(answer):
    container = st.empty()
    typed = ""
    for char in answer:
        typed += char
        container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(0.008)

# --- Handle Query ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-bubble clearfix'>{prompt}</div>", unsafe_allow_html=True)

    keywords = ["valuation", "cost", "expense", "amount", "revenue", "income", "EBITDA", "price", "value", "table", "rate", "component"]
    chain = st.session_state.qa_chain_table if any(k in prompt.lower() for k in keywords) else st.session_state.qa_chain_full

    with st.spinner("Thinking..."):
        refined_prompt = preprocess_query(prompt)
        result = chain.invoke({"query": refined_prompt})
        answer = result["result"]
        doc = result["source_documents"][0] if result["source_documents"] else None

        st.session_state.messages.append({"role": "assistant", "content": answer})
        typewriter_output(answer)

        if doc:
            page = doc.metadata.get("page_number", "Unknown")
            with st.popover("📘 Source Info"):
                st.markdown(f"Page: {page}")
                st.markdown("### 🔍 Retrieved Snippet")
                st.code(doc.page_content[:600])

                with tempfile.TemporaryDirectory() as tmp:
                    images = convert_from_path(PDF_PATH, dpi=150, first_page=page, last_page=page, output_folder=tmp)
                    if images:
                        st.image(images[0], caption=f"Page {page}", use_container_width=True)

# --- Floating Valuation Button ---
st.markdown("""
    <div class="floating-button">
        <form action="" method="post">
            <button onclick="window.location.reload();">Valuation 💰</button>
        </form>
    </div>
""", unsafe_allow_html=True)
