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
import fitz

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
        markdown_table =answer.strip().removeprefix("markdown").removesuffix("").strip()
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


# === Upload PDF ===
uploaded_pdf = st.file_uploader("Upload a Valuation PDF", type=["pdf"])
if uploaded_pdf is None:
    st.stop()

# Save uploaded PDF to a temp path
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_pdf.read())
    PDF_PATH = tmp_file.name

# Setup temp output folder
output_folder = tempfile.mkdtemp()
image_folder = output_folder
extracted_folder = output_folder  # used in parse_pdf()


# === Target Setup ===
target_headings = {
    "income_approach": {"text": "INCOME APPROACH", "take": 1},
    "market_approach": {"text": "MARKET APPROACH", "take": 2},
}

valuation_summary_text = "VALUATION SUMMARY"
valuation_summary_page = None

# === Step 1: Extract heading-based matches (skip early pages)
doc = fitz.open(PDF_PATH)
heading_pages = {key: [] for key in target_headings}
for i in range(len(doc)):
    if i < 5:
        continue
    text = doc[i].get_text().upper()
    for key, config in target_headings.items():
        if config["text"] in text:
            heading_pages[key].append(i)
    if valuation_summary_page is None and valuation_summary_text in text:
        valuation_summary_page = i
doc.close()


# === Step 2: Select exact pages
final_selections = []

for key, pages in heading_pages.items():
    take_index = target_headings[key]["take"] - 1
    if take_index < len(pages):
        selected_page = pages[take_index]
        final_selections.append((selected_page, key))

        # Add next page after market approach
        if key == "market_approach":
            total_pages = len(fitz.open(pdf_path))
            if selected_page + 1 < total_pages:
                final_selections.append((selected_page + 1, f"{key}_continued"))

# Add valuation summary if found
if valuation_summary_page is not None:
    final_selections.append((valuation_summary_page, "valuation_summary"))

# === Step 3: Convert to images
all_images = convert_from_path(PDF_PATH, dpi=300)

openai.api_key = os.getenv("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")  # Set via env variable
for idx, label in final_selections:
    image = all_images[idx]
    image_path = os.path.join(image_folder, f"{label}_page_{idx+1}.png")
    image.save(image_path)


# === OpenAI API Key ===
openai.api_key = "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"



# === Loop through each PNG file ===
for filename in sorted(os.listdir(output_folder)):
    if filename.endswith(".png"):
        image_path = os.path.join(output_folder, filename)
        print(f"\n🖼️ Processing: {filename}")

        # Extract page number from filename
        match = re.search(r'page_(\d+)', filename)
        if not match:
            print("❌ Could not extract page number. Skipping.")
            continue
        page_num = match.group(1)

        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Send to GPT-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
Can you extract all values and details precisely from the image? Return the response in a clean and readable format.

If formulas are present, write "THE EQUATION" using simple math symbols like x, -, and = (instead of technical or LaTeX format). Clean any unnecessary text or symbols.

Make sure the final output is easy to read and organized properly.
""" },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                },
            ],
            max_tokens=512,
        )

        # === Save Response to Text File ===
        output_text = response.choices[0].message.content
        output_path = os.path.join(output_folder, f"page_{page_num}.txt")

        with open(output_path, "w") as f:
            f.write(output_text)

        print(f"✅ Saved: page_{page_num}.txt")



# --- Session State ---
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.docs = []
    st.session_state.vector_store = None
    st.session_state.retriever = None
    st.session_state.messages.append({"role": "assistant", "content": "Hi! I am here to answer any questions you may have about your valuation report."})
    st.session_state.messages.append({"role": "assistant", "content": "What can I help you with?"})

if "source_infos" not in st.session_state:
    st.session_state.source_infos = []

# Step 1: Parsing Pdf
def parse_pdf():
    parser = LlamaParse(
        api_key="llx-GXPHf09BoCtf4RciC9CqmLMRvMAdMM1X6taKcwhWGKxVFP4S",
        num_workers=4
    )
    result = parser.parse("/Users/jaipdalvi/Desktop/Work/Gen AI/Langchain/Galligan Holdings Certified Valuation Report.pdf")

    pages = []
    for page in result.pages:
        page_num = page.page  # LlamaParse uses 1-based indexing

        # === Check if there's a replacement file ===
        replacement_path = os.path.join(extracted_folder, f"page_{page_num}.txt")
        if os.path.exists(replacement_path):
            with open(replacement_path, "r") as f:
                content = f.read().strip()
        else:
            # Original logic: clean parsed Markdown
            content = page.md.strip()

            # Clean unwanted "null" and empty lines
            cleaned_lines = [
                line for line in content.splitlines()
                if line.strip().lower() != "null" and line.strip() != ""
            ]
            content = "\n".join(cleaned_lines)

            # Hardcoded fix for table header if needed
            if "| Score | China Exposure" in content and "Vulnerability" not in content:
                content = content.replace("| Score |", "| Vulnerability Score |")

        if content:
            pages.append(Document(page_content=content, metadata={"page_number": page_num}))

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=3300,chunk_overlap=0,separators=["\n"])
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
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 50,"fetch_k": 100,"lambda_mult":0.9}) #k:50, fetch_k:100

        # Step 5: Add Cohere reranker
        reranker = CohereRerank(model="rerank-english-v3.0", user_agent="langchain",top_n=10) #by default top_n=3

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
if st.session_state.source_infos:
    for idx, info in enumerate(st.session_state.source_infos):
        with st.popover(f"📘 Source Info {idx+1}"):
            st.markdown(f"**Page: {info['page']}**")
            st.image(info["image"], caption=f"Page {info['page']}", use_container_width=True)


   



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

3. **Valuation method / theory / reasoning questions**
   • If the question involves **valuation methods**, **concluded value**, or topics like **Income Approach**, **Market Approach**, or **Valuation Summary**, do the following:
     - Combine and synthesize relevant information across all chunks.
     - Pay special attention to how **weights are distributed** (e.g., “50% DCF, 25% EBITDA, 25% SDE”).
     - Avoid oversimplifying if more detailed breakdowns (like subcomponents of market approach) are available.
     - If a table gives a simplified view (e.g., "50% Market Approach"), but other parts break it down (e.g., 25% EBITDA + 25% SDE), **prefer the detailed breakdown with percent value**.   
     - When describing weights, also mention the **corresponding dollar values** used in the context (e.g., “50% DCF = $3,712,000, 25% EBITDA = $4,087,000...”)
     - **If Market approach is composed of sub-methods like EBITDA and SDE, then explicitly extract and show their individual weights and values, even if not listed together in a single table.**


4. **Theory/textual question**  
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

    # --- Expand abbreviations in user question if their full forms exist in context ---
   

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
        with tempfile.TemporaryDirectory() as tmp:
            images = convert_from_path(PDF_PATH, dpi=150, first_page=page, last_page=page, output_folder=tmp)
            if images:
                st.session_state.source_infos.append({
                    "page": page,
                    "image": images[0]
                })
                st.rerun()



    

    
