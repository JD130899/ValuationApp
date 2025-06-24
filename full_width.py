import streamlit as st
import tempfile
import time
from openai import OpenAI
from llama_cloud_services import LlamaParse
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pdf2image import convert_from_path

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
            position: fixed; bottom: 80px; right: 20px; z-index: 9999;
        }
        .floating-button button {
            background-color: #444 !important;
            color: white !important; padding: 10px 18px;
            border-radius: 999px; border: none;
        }
        .small-text {
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title + Greeting ---
st.title("Underwriting Agent")

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."})
    st.session_state.messages.append({"role": "assistant", "content": "What can I help you with?"})

# --- Show Messages at Top ---
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

# --- File Upload ---
uploaded_pdf = st.file_uploader("Upload a Valuation PDF", type=["pdf"])
if uploaded_pdf is not None and "PDF_PATH" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        st.session_state.PDF_PATH = tmp_file.name

# --- Valuation Button + Chat Prompt (always visible) ---
valuation_triggered = st.button("Valuation 💰 (ask valuation)", key="valuation_btn")
user_input = st.chat_input("Message...")
prompt = "What is the valuation?" if valuation_triggered else user_input

# --- PDF Parsing Logic ---
def parse_pdf():
    parser = LlamaParse(
        api_key=st.secrets["LLAMA_CLOUD_API_KEY"],
        num_workers=4,
        verbose=False,
        language="en"
    )
    result = parser.parse(st.session_state.PDF_PATH)
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

def get_vectorstores(docs):
    embed = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    table_docs = [doc for doc in docs if doc.metadata.get("type") == "table"]
    full_vs = FAISS.from_documents(docs, embed)
    table_vs = FAISS.from_documents(table_docs, embed)
    return full_vs.as_retriever(), table_vs.as_retriever()

def get_qa_chains(full_ret, table_ret):
    llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    custom_prompt = PromptTemplate.from_template("""
You are a helpful financial assistant. Based on the context below, extract markdown tables exactly as is without any modifications or reformatting. Do not summarize. Only include the table using proper markdown format.

Context:
{context}

Question:
{question}
""")
    qa_full = RetrievalQA.from_chain_type(llm=llm, retriever=full_ret, return_source_documents=True)
    qa_table = RetrievalQA.from_chain_type(llm=llm, retriever=table_ret, return_source_documents=True, chain_type_kwargs={"prompt": custom_prompt})
    return qa_full, qa_table

# --- Run Backend on First Time PDF Upload ---
if uploaded_pdf and not st.session_state.initialized:
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

# --- Typing Effect ---
def typewriter_output(answer):
    container = st.empty()
    typed = ""
    for char in answer:
        typed += char
        container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(0.008)

# --- Chat Handling ---
if prompt and st.session_state.initialized:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-bubble clearfix'>{prompt}</div>", unsafe_allow_html=True)

    keywords = ["table", "rate", "amount", "cost", "component"]
    chain = st.session_state.qa_chain_table if any(k in prompt.lower() for k in keywords) else st.session_state.qa_chain_full

    with st.spinner("Thinking..."):
        result = chain.invoke({"query": prompt})
        answer = result["result"]
        doc = result["source_documents"][0] if result["source_documents"] else None

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # Show markdown table directly or type out normal response
        if "|" in answer and "---" in answer:
            st.markdown(answer)
        else:
            typewriter_output(answer)

        # 📘 Show Source Info
        if doc:
            page = doc.metadata.get("page_number", "Unknown")
            with st.popover("📘 Source Info"):
                st.markdown(f"<span class='small-text'>Page: {page}</span>", unsafe_allow_html=True)
                st.markdown("<span class='small-text'>**Extracted Text:**</span>", unsafe_allow_html=True)
                st.markdown(f"<span class='small-text'>{doc.page_content}</span>", unsafe_allow_html=True)

# --- Always-visible Floating Button (reload to trigger valuation) ---
st.markdown("""
    <div class="floating-button">
        <form action="" method="post">
            <button onclick="window.location.reload();">Valuation 💰</button>
        </form>
    </div>
""", unsafe_allow_html=True)
