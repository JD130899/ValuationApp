import streamlit as st
from langchain_core.documents import Document
from llama_cloud_services import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import time
import numpy as np
import os
from pdf2image import convert_from_path
import tempfile
import difflib
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
# --- Streamlit UI Config ---
st.set_page_config(page_title="Valuation RAG Chatbot", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
        .stApp {
            background-color: black;
            color: white;
        }
        .user-bubble {
            background-color: #2a2a2a;
            color: white;
            padding: 10px;
            border-radius: 12px;
            max-width: 60%;
            float: right;
            margin: 5px 0 10px auto;
            text-align: right;
        }
        .assistant-bubble {
            background-color: #1e1e1e;
            color: white;
            padding: 10px;
            border-radius: 12px;
            max-width: 60%;
            float: left;
            margin: 5px auto 10px 0;
            text-align: left;
        }
        .clearfix::after {
            content: "";
            display: block;
            clear: both;
        }
        .floating-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
        }
        .floating-button button {
            background-color: #444 !important;
            color: white !important;
            padding: 10px 18px;
            border-radius: 999px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

def typewriter_output(answer):
    if answer.strip().startswith("```markdown"):
        # Extract markdown table from triple backticks
        markdown_table = answer.strip().removeprefix("```markdown").removesuffix("```").strip()
        st.markdown(markdown_table)  # Proper markdown rendering (table)
    else:
        # Use typewriter effect for regular text responses
        container = st.empty()
        typed = ""
        for char in answer:
            typed += char
            container.markdown(f"<div class='assistant-bubble clearfix'>{typed}</div>", unsafe_allow_html=True)
            time.sleep(0.008)


# --- Title ---
st.title("Underwriting Agent")


# --- PDF Upload Logic ---
uploaded_pdf = st.file_uploader("Upload a Valuation PDF", type=["pdf"])
if uploaded_pdf is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        PDF_PATH = tmp_file.name
else:
    st.stop()

# --- Session State ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.docs = []
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."})
    st.session_state.messages.append({"role": "assistant", "content": "What can I help you with?"})


# Step 1: Parsing Pdf
def parse_pdf():
    parser = LlamaParse(api_key="llx-GXPHf09BoCtf4RciC9CqmLMRvMAdMM1X6taKcwhWGKxVFP4S", num_workers=4)
    result = parser.parse(PDF_PATH)
    
    pages = []
    for page in result.pages:
        content = page.md.strip()
        
        # Clean unwanted "null" and empty lines
        cleaned_lines = [
            line for line in content.splitlines()
            if line.strip().lower() != "null" and line.strip() != ""
        ]
        
        cleaned_content = "\n".join(cleaned_lines)

        # Hard coding   
        if "| Score | China Exposure" in cleaned_content and "Vulnerability" not in cleaned_content:
            cleaned_content = cleaned_content.replace("| Score |", "| Vulnerability Score |")
        
        if cleaned_content:
            pages.append(Document(page_content=cleaned_content, metadata={"page_number": page.page}))
    
    return pages





def find_best_matching_doc_reranker(answer_text, retrieved_docs, reranker):
    # Rerank based on similarity between the model's answer and each doc
    reranked = reranker.compress_documents(retrieved_docs, query=answer_text)
    return reranked[0] if reranked else None




user_question = st.chat_input("Message...")

# --- Initial Load ---
if not st.session_state.initialized:
    with st.spinner("Parsing PDF..."):
        docs = parse_pdf()

    with st.spinner("Chunking content..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=0,separators=["\n"])
        split_docs = splitter.split_documents(docs)
    
        for idx, doc in enumerate(split_docs, start=1):
            doc.metadata["chunk_id"] = idx        
            page = doc.metadata.get("page_number", "-")

    with st.spinner("Building vectorstore..."):
        os.environ["COHERE_API_KEY"] = "qeNFbHVCZhyb1pmOhlcNYIAMItuV4xCOQwk4OSq0"
        embedding = CohereEmbeddings(model="embed-english-v3.0", user_agent="langchain")

        # Step 3: Create FAISS vector store
        vectorstore = FAISS.from_documents(split_docs, embedding)

        # Step 4: Create MMR Retriever from FAISS
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 25,"fetch_k": 30,"lambda_mult":0.9}) #k:50, fetch_k:100

        # Step 5: Add Cohere reranker
        reranker = CohereRerank(model="rerank-english-v3.0", user_agent="langchain") #by default top_n=3

        # Step 6: Wrap in ContextualCompressionRetriever
        final_retriever = ContextualCompressionRetriever(
            base_retriever=retriever,
            base_compressor=reranker,
        )


        

    st.session_state.docs = docs
    st.session_state.vector_store = vectorstore
    st.session_state.retriever = final_retriever
    st.session_state.reranker = reranker  # ✅ store for later use
    st.session_state.initialized = True


# --- Show chat history ---
for msg in st.session_state.messages:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    if msg["content"].strip().startswith("```markdown"):
    # Render markdown table properly
        markdown_table = msg["content"].strip().removeprefix("```markdown").removesuffix("```").strip()
        st.markdown(markdown_table)
    else:
        # Render regular bubble
        st.markdown(f"<div class='{role_class} clearfix'>{msg['content']}</div>", unsafe_allow_html=True)

# --- Persistent Source Info display ---
if "source_image" in st.session_state:
    with st.popover("📘 Source Info"):
        st.markdown(f"**Page: {st.session_state.source_page}**")
        st.image(st.session_state.source_image, caption=f"Page {st.session_state.source_page}", use_container_width=True)






# --- Answer logic ---
if user_question:
    with st.spinner("Retrieving context..."):
        retrieved_docs = st.session_state.retriever.invoke(user_question)
        print(retrieved_docs)
        for doc in retrieved_docs:
            print(doc.page_content)
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
    template = """
    You are a financial-data extraction assistant.

    **Use ONLY what appears under “Context”.**

    ### How to answer
    1. **Single value questions**  
    • Find the row + column that match the user's words.  
    • Return the answer in a **short, clear sentence** using the exact number from the context.  
        Example: “The Income (DCF) approach value is $1,150,000.”  
    • **Do NOT repeat the metric name or company name** unless the user asks.

    2. **Table questions**  
    • Return the full table **with its header row** in GitHub-flavoured markdown.

    3. **Theory/textual question**  
    • Try to return an explanation **based on the context**.

    If you still cannot see the answer, reply **“I don't know.”**

    ---
    Context:
    {context}

    ---
    Question: {question}
    Answer:""",
        input_variables=["context", "question"]
    )

    final_prompt = prompt.invoke({"context": context_text, "question": user_question})

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(f"<div class='user-bubble clearfix'>{user_question}</div>", unsafe_allow_html=True)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    #with st.spinner("Thinking..."):
    response = llm.invoke(final_prompt)
    #st.markdown(f"<div class='assistant-bubble clearfix'>{response.content}</div>", unsafe_allow_html=True)

    typewriter_output(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})

    
    matched_doc = find_best_matching_doc_reranker(response.content, retrieved_docs, st.session_state.reranker)

    if matched_doc:
        page = matched_doc.metadata.get("page_number", "Unknown")
        with st.popover("📘 Source Info"):
            st.markdown(f"**Page: {page}**")
            with tempfile.TemporaryDirectory() as tmp:
                images = convert_from_path(PDF_PATH, dpi=150, first_page=page, last_page=page, output_folder=tmp)
                if images:
                    st.image(images[0], caption=f"Page {page}", use_container_width=True)
                    st.session_state.source_image = images[0]
                    st.session_state.source_page = page


    

    
