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

# --- Session State Setup ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."})
    st.session_state.messages.append({"role": "assistant", "content": "What can I help you with?"})

# --- File Upload ---
uploaded_pdf = st.file_uploader("Upload a Valuation PDF", type=["pdf"])
if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        PDF_PATH = tmp_file.name
else:
    st.stop()


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
            background-color: #444 !important;
            color: white !important; padding: 10px 18px;
            border-radius: 999px; border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("Underwriting Agent")


# --- Show Intro Messages Early ---
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)
# --- Button Trigger ---
#valuation_clicked = st.empty()
#valuation_triggered = valuation_clicked.button("Valuation 💰", key="valuation_btn", help="Click to ask about valuation")
user_input = st.chat_input("Message...")
prompt = user_input

# --- PDF Parsing ---
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

# --- Embeddings + Vectorstores ---
def get_vectorstores(docs):
    embed = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    table_docs = [doc for doc in docs if doc.metadata.get("type") == "table"]
    full_vs = FAISS.from_documents(docs, embed)
    table_vs = FAISS.from_documents(table_docs, embed)
    return full_vs.as_retriever(), table_vs.as_retriever()

# --- QA Chains ---
def get_qa_chains(full_ret, table_ret):
    llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    custom_prompt = PromptTemplate.from_template("""
You are a helpful financial assistant. Based on the context below, extract markdown tables exactly as is without any modifications or reformatting. Do not summarize. Only include the table using proper markdown format.

Context:
{context}

Question:
{question}
""")
    qa_full = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=full_ret,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

    qa_table = RetrievalQA.from_chain_type(llm=llm, retriever=table_ret, return_source_documents=True, chain_type_kwargs={"prompt": custom_prompt})
    return qa_full, qa_table

# --- First-Time Setup ---
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



# --- Typing Effect ---
def typewriter_output(answer):
    container = st.empty()
    typed = ""
    for char in answer:
        typed += char
        container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
        time.sleep(0.008)

# --- Chat Handling ---
from markdown import markdown as md_to_html
import html

# --- Chat Handling ---
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='user-bubble clearfix'>{prompt}</div>", unsafe_allow_html=True)

    keywords = ["table", "rate", "amount", "cost", "component"]
    chain = st.session_state.qa_chain_table if any(k in prompt.lower() for k in keywords) else st.session_state.qa_chain_full

    with st.spinner("Thinking..."):
        result = chain.invoke({"query": prompt})
        answer = result["result"]
        doc = result["source_documents"][0] if result["source_documents"] else None

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 1. Strip triple backticks
        if answer.strip().startswith("```") and answer.strip().endswith("```"):
            answer = answer.strip().strip("`").replace("markdown", "").strip()

        # 2. Convert markdown (with tables) to HTML
        html_answer = md_to_html(answer, extensions=['tables', 'extra'])

        # 3. Wrap in styled assistant bubble
        wrapped = f"<div class='assistant-bubble clearfix'>{html_answer}</div>"
        st.markdown(wrapped, unsafe_allow_html=True)

        # Optional: Source info
        if doc:
            page = doc.metadata.get("page_number", "Unknown")
            with st.popover("📘 Source Info"):
                st.markdown(f"Page: {page}")
                st.markdown("*Extracted Text:*")
                st.markdown(doc.page_content)



# --- Floating Valuation Button ---
#st.markdown("""
#   <div class="floating-button">
#      <form action="" method="post">
#            <button onclick="window.location.reload();">Valuation 💰</button>
#        </form>
#    </div>
#""", unsafe_allow_html=True)
