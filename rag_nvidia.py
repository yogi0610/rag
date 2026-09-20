import os
import hashlib
import pickle
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Safe imports for Hybrid Search to support both legacy and modular LangChain versions
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    BM25Retriever = None

try:
    # pyrefly: ignore [missing-import]
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from langchain_classic.retrievers import EnsembleRetriever
    except ImportError:
        EnsembleRetriever = None


# Robust Pure-Python Reciprocal Rank Fusion (RRF) for Hybrid Search
def reciprocal_rank_fusion(results_list: list, k: int = 60):
    """Combines multiple ranked document lists using Reciprocal Rank Fusion."""
    fused_scores = {}
    doc_map = {}
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = getattr(doc, "page_content", str(doc))
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1.0 / (rank + k)
    sorted_doc_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]


class SimpleHybridRetriever:
    """Robust hybrid retriever combining BM25 keyword matching with FAISS vector search."""
    def __init__(self, bm25_retriever, faiss_retriever, top_k: int = 8):
        self.bm25_retriever = bm25_retriever
        self.faiss_retriever = faiss_retriever
        self.top_k = top_k

    def invoke(self, query: str):
        bm25_docs = []
        faiss_docs = []
        if self.bm25_retriever is not None:
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception:
                bm25_docs = []
        if self.faiss_retriever is not None:
            try:
                faiss_docs = self.faiss_retriever.invoke(query)
            except Exception:
                faiss_docs = []
        if bm25_docs and faiss_docs:
            fused = reciprocal_rank_fusion([bm25_docs, faiss_docs])
            return fused[:self.top_k]
        return (faiss_docs or bm25_docs)[:self.top_k]


# 1. Page Configuration (No Sidebar)
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Universal default parameters tuned for all conditions
MODEL_NAME = "meta/muse-glimmer-30b"
USE_HYBRID = True
USE_RERANKER = True
TOP_K = 4

# Cache directory for persistent vector store
CACHE_DIR = ".faiss_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# 2. Secure API Key Resolution
api_key = os.getenv("NVIDIA_API_KEY") or (
    st.secrets.get("NVIDIA_API_KEY") if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets else None
)

if not api_key:
    st.title("📄 PDF RAG Assistant")
    api_key = st.text_input(
        "Enter NVIDIA API Key to continue:",
        type="password",
        placeholder="nvapi-...",
        help="Add NVIDIA_API_KEY to your .env or Streamlit Secrets to skip this step."
    )
    if not api_key:
        st.info("ℹ️ Please provide an NVIDIA API key to start.")
        st.stop()

# 3. Model Initializers (Cached)
@st.cache_resource(show_spinner=False)
def get_embeddings(key: str):
    return NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        api_key=key
    )

@st.cache_resource(show_spinner=False)
def get_llm(key: str):
    return ChatNVIDIA(
        model=MODEL_NAME,
        api_key=key,
        temperature=0.1,
        max_tokens=1500,
        timeout=120
    )

@st.cache_resource(show_spinner=False)
def get_reranker(key: str):
    try:
        # pyrefly: ignore [missing-import]
        from langchain_nvidia_ai_endpoints import NVIDIARerank
        return NVIDIARerank(
            model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
            api_key=key,
            top_n=TOP_K
        )
    except Exception:
        return None

embeddings = get_embeddings(api_key)
llm = get_llm(api_key)
reranker = get_reranker(api_key) if USE_RERANKER else None

# 4. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []

if "current_file_hash" not in st.session_state:
    st.session_state.current_file_hash = None

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary = None

# 5. Header & Multi-PDF Ingestion Bar
st.title("📄 PDF Assistant")

def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()

# Top control toolbar
upload_col, btn_col1, btn_col2 = st.columns([3, 1, 1])

with upload_col:
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

with btn_col1:
    if uploaded_files and st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

with btn_col2:
    if uploaded_files and st.button("📑 Summary", use_container_width=True):
        if st.session_state.doc_chunks:
            with st.spinner("Generating document summary..."):
                try:
                    sample_text = "\n\n".join([
                        f"[{c.metadata.get('filename', 'Doc')}]: {c.page_content}"
                        for c in st.session_state.doc_chunks[:8]
                    ])
                    summary_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an expert analyst. Provide a structured summary of the uploaded document(s) and suggest 3 insightful questions."),
                        ("human", "Documents preview:\n{text}")
                    ])
                    summary_chain = summary_prompt | llm | StrOutputParser()
                    st.session_state.doc_summary = summary_chain.invoke({"text": sample_text})
                except Exception as e:
                    st.error(f"⚠️ Could not generate summary (NVIDIA API: {e})")

# Ingest and Index Multiple PDF Documents
if uploaded_files:
    # Compute combined hash for the collection of files
    combined_hash_input = "".join(sorted([f.name + get_file_hash(f.getvalue()) for f in uploaded_files]))
    combined_hash = hashlib.md5(combined_hash_input.encode()).hexdigest()
    collection_cache_path = os.path.join(CACHE_DIR, combined_hash)

    if st.session_state.current_file_hash != combined_hash:
        st.session_state.current_file_hash = combined_hash
        st.session_state.messages = []
        st.session_state.doc_summary = None

        all_chunks = []
        loaded_from_cache = False

        # Attempt to load combined collection from cache
        if os.path.exists(collection_cache_path) and os.path.exists(os.path.join(collection_cache_path, "chunks.pkl")):
            try:
                with st.spinner("Loading cached index from disk..."):
                    vectorstore = FAISS.load_local(
                        collection_cache_path,
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    with open(os.path.join(collection_cache_path, "chunks.pkl"), "rb") as f:
                        all_chunks = pickle.load(f)
                    loaded_from_cache = True
            except Exception:
                pass

        if not loaded_from_cache:
            with st.spinner(f"Indexing {len(uploaded_files)} PDF document(s)..."):
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\nQuestion", "\n\nQ", "\n\n", "\n", " ", ""]
                )

                for f in uploaded_files:
                    f_bytes = f.getvalue()
                    f_hash = get_file_hash(f_bytes)
                    single_cache = os.path.join(CACHE_DIR, f_hash)

                    file_chunks = []
                    # Check individual file cache
                    if os.path.exists(single_cache) and os.path.exists(os.path.join(single_cache, "chunks.pkl")):
                        try:
                            with open(os.path.join(single_cache, "chunks.pkl"), "rb") as pkl:
                                file_chunks = pickle.load(pkl)
                        except Exception:
                            file_chunks = []

                    if not file_chunks:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(f_bytes)
                            tmp_file_path = tmp_file.name

                        try:
                            loader = PyPDFLoader(tmp_file_path)
                            docs = loader.load()
                            for doc in docs:
                                doc.metadata["filename"] = f.name

                            file_chunks = text_splitter.split_documents(docs)

                            for i, chunk in enumerate(file_chunks):
                                chunk.metadata["chunk_id"] = i
                                chunk.metadata["filename"] = f.name

                            os.makedirs(single_cache, exist_ok=True)
                            with open(os.path.join(single_cache, "chunks.pkl"), "wb") as pkl:
                                pickle.dump(file_chunks, pkl)
                        finally:
                            if os.path.exists(tmp_file_path):
                                os.remove(tmp_file_path)

                    all_chunks.extend(file_chunks)

                # Build combined FAISS index and cache it
                vectorstore = FAISS.from_documents(all_chunks, embeddings)
                vectorstore.save_local(collection_cache_path)

                with open(os.path.join(collection_cache_path, "chunks.pkl"), "wb") as pkl:
                    pickle.dump(all_chunks, pkl)

        st.session_state.doc_chunks = all_chunks

        # Build Hybrid Retriever
        faiss_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K * 2 if USE_RERANKER else TOP_K}
        )

        if USE_HYBRID and all_chunks and BM25Retriever is not None:
            try:
                bm25_retriever = BM25Retriever.from_documents(all_chunks)
                bm25_retriever.k = TOP_K * 2 if USE_RERANKER else TOP_K
                
                if EnsembleRetriever is not None:
                    st.session_state.retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever],
                        weights=[0.4, 0.6]
                    )
                else:
                    st.session_state.retriever = SimpleHybridRetriever(
                        bm25_retriever=bm25_retriever,
                        faiss_retriever=faiss_retriever,
                        top_k=TOP_K * 2 if USE_RERANKER else TOP_K
                    )
            except Exception:
                st.session_state.retriever = faiss_retriever
        else:
            st.session_state.retriever = faiss_retriever

        status_text = "⚡ Loaded from cache" if loaded_from_cache else "🔨 Indexed collection"
        st.toast(f"{status_text}: {len(all_chunks)} chunks across {len(uploaded_files)} PDF(s)", icon="✅")

# If no files uploaded yet, show welcome instructions
if not uploaded_files or st.session_state.retriever is None:
    st.info("👆 Please upload one or more PDF documents above to begin asking questions.")
    st.stop()

# 6. Executive Summary Card (if generated)
if st.session_state.doc_summary:
    with st.expander("📑 Document Executive Summary & Suggested Questions", expanded=True):
        st.markdown(st.session_state.doc_summary)

# 7. Conversational RAG Pipeline
def format_docs(docs):
    formatted = []
    for d in docs:
        doc_name = d.metadata.get("filename", "Document")
        page = d.metadata.get("page", 0)
        page_num = page + 1 if isinstance(page, int) else page
        formatted.append(f"[{doc_name} — Page {page_num}]:\n{d.page_content.strip()}")
    return "\n\n".join(formatted)

def retrieve_and_rerank(query: str, base_retriever, reranker_model, final_k: int):
    try:
        initial_docs = base_retriever.invoke(query)
    except Exception:
        initial_docs = []

    if reranker_model and initial_docs:
        try:
            compressed = reranker_model.compress_documents(query=query, documents=initial_docs)
            return compressed[:final_k]
        except Exception:
            return initial_docs[:final_k]
    return initial_docs[:final_k]

# Conversational Reformulation
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question that can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if needed "
        "and otherwise return it as is."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

# Grounded QA Chain
qa_system_prompt = (
    "You are an expert technical assistant. Answer the user's question "
    "using ONLY the facts, programming questions, constraints, and test cases provided "
    "in the context below. When referencing information, mention which document and page it came from.\n\n"
    "STRICT RULES:\n"
    "1. Do not fabricate, assume, or infer details not explicitly stated in the context.\n"
    "2. If the context does not contain the answer, reply: 'The provided document(s) do not contain sufficient information to answer this question.'\n"
    "3. Maintain exact fidelity to code logic, variables, inputs, and outputs.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])
qa_chain = qa_prompt | llm | StrOutputParser()

# 8. Chat History Display
langchain_history = []
for msg in st.session_state.messages:
    if msg["role"] == "user":
        langchain_history.append(HumanMessage(content=msg["content"]))
    else:
        langchain_history.append(AIMessage(content=msg["content"]))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander(f"🔍 Sources ({len(msg['sources'])})", expanded=False):
                for idx, src in enumerate(msg["sources"], start=1):
                    doc_name = src.metadata.get("filename", "Document")
                    page = src.metadata.get("page", 0)
                    page_num = page + 1 if isinstance(page, int) else page
                    st.markdown(f"**Source #{idx} — `{doc_name}` (Page {page_num})**")
                    st.code(src.page_content, language="text")

# 9. User Input & Streaming Generation
doc_count = len(uploaded_files)
input_placeholder = (
    f"Ask about '{uploaded_files[0].name}'..."
    if doc_count == 1
    else f"Ask across {doc_count} uploaded documents..."
)

if user_query := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        # Step A: Contextualize if multi-turn history exists
        if len(langchain_history) > 0:
            with st.spinner("Understanding conversation context..."):
                standalone_query = contextualize_chain.invoke({
                    "chat_history": langchain_history,
                    "question": user_query
                })
        else:
            standalone_query = user_query

        # Step B: Retrieve + Rerank
        with st.spinner("Searching across documents..."):
            retrieved_docs = retrieve_and_rerank(
                query=standalone_query,
                base_retriever=st.session_state.retriever,
                reranker_model=reranker if USE_RERANKER else None,
                final_k=TOP_K
            )
            context_text = format_docs(retrieved_docs)

        # Step C: Stream Answer
        full_answer = None
        try:
            stream = qa_chain.stream({
                "context": context_text,
                "chat_history": langchain_history,
                "question": user_query
            })
            full_answer = st.write_stream(stream)
        except Exception as e:
            err_msg = str(e)
            if "timeout" in err_msg.lower():
                st.error(
                    f"⏱️ **NVIDIA NIM Timeout**: The model `{MODEL_NAME}` took longer than 120 seconds to respond. "
                    "This typically occurs when NVIDIA's hosted endpoints for this model are under heavy load or queuing requests. "
                    "Please try your question again, or switch to a high-availability model such as `google/gemma-4-31b-it` or `meta/llama-3.3-70b-instruct`."
                )
            else:
                st.error(f"⚠️ **Error generating response**: {err_msg}")

        # Step D: Citations with Document Names
        if full_answer:
            if retrieved_docs:
                with st.expander(f"🔍 Sources ({len(retrieved_docs)})", expanded=False):
                    for idx, doc in enumerate(retrieved_docs, start=1):
                        doc_name = doc.metadata.get("filename", "Document")
                        page = doc.metadata.get("page", 0)
                        page_num = page + 1 if isinstance(page, int) else page
                        st.markdown(f"**Source #{idx} — `{doc_name}` (Page {page_num})**")
                        st.code(doc.page_content, language="text")

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": retrieved_docs
            })
