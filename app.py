# Packages
import streamlit as st
from PyPDF2 import PdfReader
import os
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

# Page Config (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Cognitive PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Pure Advanced CSS - No HTML Layout elements)
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        /* Global Page Styling */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Outfit', sans-serif !important;
            background: linear-gradient(135deg, #0d0a1b 0%, #150f2e 50%, #080512 100%) !important;
            color: #f1f0f7 !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(16, 12, 34, 0.95) !important;
            border-right: 1px solid rgba(123, 97, 255, 0.15) !important;
        }
        
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: #bfaaff !important;
            font-weight: 600 !important;
        }

        /* Glass Cards via Container Keys */
        .st-key-auth-card, .st-key-upload-card, .st-key-chat-card, .st-key-control-card {
            background: rgba(255, 255, 255, 0.02) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            padding: 20px !important;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            margin-bottom: 20px !important;
        }
        
        /* App Title Container Styling */
        .st-key-app-title-container h1 {
            text-align: center !important;
            background: linear-gradient(90deg, #bfaaff, #8c7ae6, #ff7675) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-weight: 700 !important;
            margin-bottom: 0px !important;
        }
        .st-key-app-title-container p {
            text-align: center !important;
            color: #a29bfe !important;
            font-size: 16px !important;
            margin-top: 5px !important;
            margin-bottom: 30px !important;
        }
        
        /* User Chat Message Bubble Styling (Wildcard matches index keys) */
        div[class*="st-key-user-msg-"] [data-testid="stChatMessage"] {
            background: linear-gradient(135deg, #6c5ce7 0%, #8c7ae6 100%) !important;
            color: #ffffff !important;
            border-radius: 14px 14px 4px 14px !important;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.2) !important;
            margin-left: auto !important;
            max-width: 85% !important;
            border: none !important;
        }

        div[class*="st-key-user-msg-"] [data-testid="stChatMessage"] * {
            color: #ffffff !important;
        }

        /* Assistant Chat Message Bubble Styling */
        div[class*="st-key-assistant-msg-"] [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            color: #f1f0f7 !important;
            border-radius: 14px 14px 14px 4px !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
            max-width: 85% !important;
        }

        div[class*="st-key-assistant-msg-"] [data-testid="stChatMessage"] * {
            color: #f1f0f7 !important;
        }
        
        /* Message Meta Info (Caption inside assistant message wrapper) */
        div[class*="st-key-assistant-msg-"] [data-testid="stCaptionContainer"] {
            color: rgba(255, 255, 255, 0.5) !important;
            font-size: 11px !important;
            border-top: 1px solid rgba(255, 255, 255, 0.07) !important;
            padding-top: 6px !important;
            margin-top: 8px !important;
            display: inline-block !important;
            width: 100% !important;
        }
        
        /* Dropzone visual check */
        [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(255, 255, 255, 0.01) !important;
            border: 1px dashed rgba(123, 97, 255, 0.2) !important;
            border-radius: 10px !important;
        }
        
        [data-testid="stFileUploaderDropzone"] button {
            background-color: rgba(255, 255, 255, 0.07) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            box-shadow: none !important;
            border-radius: 8px !important;
            width: auto !important;
        }
        
        [data-testid="stFileUploaderDropzone"] button:hover {
            background-color: rgba(255, 255, 255, 0.12) !important;
            border-color: rgba(123, 97, 255, 0.4) !important;
        }

        /* Streamlit Input & Chat Input Enhancements */
        div[data-baseweb="base-input"], div[data-baseweb="input"], [data-testid="stChatInput"] textarea {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 10px !important;
        }
        
        div[data-baseweb="base-input"] input, div[data-baseweb="input"] input, [data-testid="stChatInput"] textarea {
            color: #ffffff !important;
            background-color: transparent !important;
        }
        
        div[data-baseweb="input"]:focus-within, [data-testid="stChatInput"] textarea:focus {
            border-color: #8c7ae6 !important;
        }
        
        /* Make Streamlit Header Transparent */
        header[data-testid="stHeader"] {
            background-color: rgba(0, 0, 0, 0) !important;
        }
        
        /* Custom Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #6c5ce7 0%, #5243c2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 8px 24px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(108, 92, 231, 0.3) !important;
            width: 100%;
        }
        
        .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(108, 92, 231, 0.5) !important;
            border: none !important;
        }
        
        .stButton>button:active {
            transform: translateY(1px) !important;
        }
        
        /* Success/Info Message Customization */
        div[data-testid="stNotification"] {
            background-color: rgba(20, 15, 38, 0.8) !important;
            border: 1px solid rgba(123, 97, 255, 0.2) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Error reading PDF '{pdf.name}': {str(e)}")
    return text

# Split text into chunks
def get_text_chunks(text, model_name):
    chunk_size = 1000
    chunk_overlap = 150
    
    if model_name.lower() == "google ai":
        chunk_size = 1000
        chunk_overlap = 150
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Embed chunks and save index locally
def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name.lower() == "google ai":
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=api_key
            )
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return vector_store
        except Exception as e:
            st.error(f"Embedding generation failed: {str(e)}")
            return None
    return None

# Load the LangChain QA chain
def get_conversational_chain(model_name, api_key=None):
    if model_name.lower() == "google ai":
        prompt_template = """Answer the question as detailed as possible based on the context.
If the answer is not present in the context, reply that "the answer is not available in the context." Make sure to provide all the details with proper structure, don't answer wrongly.

Context:
{context}

Question:
{question}

Answer:"""
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                google_api_key=api_key
            )
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            return chain
        except Exception as e:
            st.error(f"Failed to create conversational chain: {str(e)}")
            return None
    return None

# Handle user queries
def user_input(user_question, model_name, api_key):
    if not api_key:
        st.warning("Please enter your Google API key in the sidebar.")
        return

    vector_store = st.session_state.get("vector_store", None)
    
    if vector_store is None:
        if os.path.exists("faiss_index"):
            try:
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key
                )
                vector_store = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                st.session_state["vector_store"] = vector_store
            except Exception as e:
                st.error(f"Error loading index from disk: {str(e)}")
                return
        else:
            st.warning("No processed index found. Please upload PDFs and click 'Process Documents' first.")
            return

    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain(model_name, api_key=api_key)
        if not chain:
            return
            
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"input_documents": docs, "question": user_question}
            )
        
        response_output = response.get("output_text", "No response returned.")
        pdf_names = st.session_state.get("pdf_names", [])
        
        st.session_state["conversation_history"].append({
            "question": user_question,
            "response": response_output,
            "pdf_names": pdf_names,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": "Gemini-1.5-Flash"
        })
    except Exception as e:
        st.error(f"Error query response: {str(e)}")

# Main app entrypoint
def main():
    inject_custom_css()
    
    # Initialize session state keys
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "vector_store" not in st.session_state:
        st.session_state["vector_store"] = None
    if "pdf_names" not in st.session_state:
        st.session_state["pdf_names"] = []

    # App Header Container (Native title & caption inside styling key)
    with st.container(key="app-title-container"):
        st.title("📄 Cognitive PDF Chatbot")
        st.caption("Instantly extract knowledge, ask questions, and chat with your PDF documents.")

    # Sidebar layout (Native elements wrapped inside custom containers for CSS card styles)
    with st.sidebar:
        with st.container(key="auth-card"):
            st.subheader("🔑 Authentication")
            api_key = st.text_input(
                "Google API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Your key remains secure and is never stored on disk."
            )
            
            # Check environment fallback
            if not api_key:
                env_key = os.getenv("GOOGLE_API_KEY", "")
                if env_key:
                    api_key = env_key
                    st.caption("✅ Using environment variable API key.")
        
        with st.container(key="upload-card"):
            st.subheader("📂 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload PDF Files",
                accept_multiple_files=True,
                type=["pdf"],
                help="Drag and drop or upload multiple PDF documents."
            )
            
            if st.button("Process Documents"):
                if not api_key:
                    st.error("Please enter a Google API Key.")
                elif not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                else:
                    with st.spinner("Extracting text and building vector database..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if not raw_text.strip():
                            st.error("No readable text could be extracted from the uploaded PDFs.")
                        else:
                            chunks = get_text_chunks(raw_text, "google ai")
                            vector_store = get_vector_store(chunks, "google ai", api_key)
                            if vector_store:
                                st.session_state["vector_store"] = vector_store
                                st.session_state["pdf_names"] = [pdf.name for pdf in pdf_docs]
                                st.success(f"Successfully processed {len(pdf_docs)} file(s)!")
        
        # Helper controls
        if st.session_state["conversation_history"]:
            with st.container(key="control-card"):
                if st.button("Clear Chat History"):
                    st.session_state["conversation_history"] = []
                    st.rerun()

    # Main Panel Chat interface (Pure Streamlit container + native chat components)
    with st.container(key="chat-card"):
        st.subheader("💬 AI Chat Assistant")
        user_question = st.chat_input("Ask something about the uploaded documents...")
        
    if user_question:
        user_input(user_question, "google ai", api_key)

    # Display conversation history (Pure Streamlit chat elements with index-based styling wrapper)
    if st.session_state["conversation_history"]:
        st.subheader("📜 Chat Log")
        for idx, chat in enumerate(reversed(st.session_state["conversation_history"])):
            pdf_info = ", ".join(chat['pdf_names']) if chat['pdf_names'] else "Unknown PDF"
            
            with st.container(key=f"user-msg-{idx}"):
                with st.chat_message("user"):
                    st.markdown(chat['question'])
                    
            with st.container(key=f"assistant-msg-{idx}"):
                with st.chat_message("assistant"):
                    st.markdown(chat['response'])
                    st.caption(f"🕒 {chat['timestamp']}  |  🤖 {chat['model_name']}  |  📂 {pdf_info}")

if __name__ == "__main__":
    main()