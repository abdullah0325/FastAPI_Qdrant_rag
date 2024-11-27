from langchain_qdrant import QdrantVectorStore
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

# Configuration
PDF_PATH = r"E:\my ai apps\FastAPI_Qdrant_rag\files\Muhammad Abdullah_CV.pdf"
OPENAI_API_KEY =st.secrets["OPENAI_API_KEY"] 
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_KEY = st.secrets["QDRANT_KEY"]
COLLECTION_NAME = st.secrets["COLLECTION_NAME"]
LLM_NAME = st.secrets["MODEL_NAME"]

# Initialize components
embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(pages)

Qdrant = QdrantVectorStore.from_documents(
    splits,
    embed_model,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_KEY,
    collection_name=COLLECTION_NAME
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt setup
prompt_str = """
Answer the user question based only on the following context:
{context} 

Question: {question}    
"""
_prompt = ChatPromptTemplate.from_template(prompt_str)
num_chunks = 3
retriever = Qdrant.as_retriever(
    search_type="similarity",
    search_kwargs={"k": num_chunks}
)

chat_llm = ChatOpenAI(
    model=LLM_NAME,
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
)

query_fetcher = itemgetter("question")
setup = {
    "question": query_fetcher,
    "context": query_fetcher | retriever | format_docs
}
_chain = setup | _prompt | chat_llm
