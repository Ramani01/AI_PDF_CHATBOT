# 📄 Cognitive PDF Chatbot

An AI-powered document assistant that extracts knowledge, indexes pages, and lets you chat with one or multiple PDF documents in real-time. Built with a premium, responsive **Dark Glassmorphism** user interface.

Powered by **Google Gemini (Gemini 1.5 Flash)** and **LangChain** with local **FAISS** vector storage.

---

## ✨ Features

- **Document Center**: Drag-and-drop file uploader supporting multiple PDF uploads simultaneously.
- **Smart Chunking & Vector Search**: Text is split into semantically sound chunks and indexed locally using FAISS to avoid repetitive embedding calls.
- **Gemini Chat Engine**: Converse naturally with your documents using an optimized LangChain QA chain.
- **Sleek Dark Theme**: Standard Streamlit components styled via advanced CSS overrides featuring glassmorphic panels, curved boundaries, and responsive gradients.
- **Built-in Developer Plugin**: Pre-configured with the **Ponytail** ruleset to enforce high-quality, minimal code solutions.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ramani01/AI_PDF_CHATBOT.git
cd AI_PDF_CHATBOT
```

### 2. Set Up a Virtual Environment
```powershell
# Create the environment
python -m venv myenv

# Activate on Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# Activate on macOS/Linux
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

---

## 🔑 Authentication

The application requires a **Google Gemini API Key** to generate embeddings and run prompts. You can either:
- Paste it securely in the **🔑 Authentication** input box in the sidebar (remains private and is never stored on disk).
- Or, pre-configure it as an environment variable:
  ```bash
  export GOOGLE_API_KEY="your-gemini-api-key"
  ```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** - For the frontend dashboard.
- **[LangChain](https://www.langchain.com/)** - Orchestrator for QA chain and PDF processing.
- **[Google Generative AI](https://ai.google.dev/)** - Embedding model (`embedding-001`) and LLM (`gemini-1.5-flash`).
- **[FAISS](https://github.com/facebookresearch/faiss)** - High-efficiency local vector index.
- **[PyPDF2](https://pypi.org/project/PyPDF2/)** - Text extraction from PDF documents.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
