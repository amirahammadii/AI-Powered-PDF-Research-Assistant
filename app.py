import os
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def load_pdf_text(file_path):
    pdf_reader = PyPDF2.PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

def run_query(vectorstore, query):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(query)

if __name__ == "__main__":
    print("🔍 Loading PDF...")
    text = load_pdf_text("A_Comprehensive_Survey_of_Deep_Learning.pdf")

    print("📄 Splitting text into chunks...")
    docs = split_text(text)

    print("📦 Creating vectorstore index...")
    vectordb = create_vectorstore(docs)

    while True:
        user_query = input("\nAsk a question about the document (or 'exit'): ")
        if user_query.lower() in ["exit", "quit"]:
            break
        print("🤖 Answer:")
        print(run_query(vectordb, user_query))
