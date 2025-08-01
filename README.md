#  LangChain PDF Research Assistant

A local AI assistant that lets you ask questions about your own PDFs — powered by LangChain, OpenAI, and Python.

##  Features
- Upload any PDF document (e.g., research papers, legal contracts)
- Use LLMs to answer questions based only on your file content
- Built with:
  - LangChain
  - FAISS (or Chroma)
  - OpenAI API
  - Python

##  Tech Stack
- Python 3.10+
- LangChain
- OpenAI
- PyPDF2 / pdfminer / unstructured
- FAISS or ChromaDB

##  How It Works
1. Load and clean the PDF text
2. Split it into manageable chunks
3. Embed and store the chunks in a vector database
4. Run RAG: take user question → find most relevant chunks → answer via LLM

##  Setup

```bash
git clone https://github.com/yourusername/langchain-pdf-assistant.git
cd langchain-pdf-assistant
pip install -r requirements.txt
