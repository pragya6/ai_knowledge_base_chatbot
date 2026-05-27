__import__('pysqlite3') # Fix for ChromaDB on streamlit, Newer version of SQLLite supported by ChromaDB
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# AI Knowledge Base Chatbot

import os
import logging
import chromadb
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, OnlinePDFLoader

# load_dotenv() # local path to api key

# For Streamlit Cloud
if "GOOGLE_API_KEY" not in os.environ:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 

# Setup logger
logging.basicConfig(level=logging.WARNING, filename="errors.log")
logger = logging.getLogger(__name__)


# Get the pdf contents with help of tempfile
def get_pdf_contents(data):
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(data)
    tmp_path = tmp.name

  docs = PyPDFLoader(tmp_path).load()

  # unlink from OS
  os.unlink(tmp_path)

  return docs



# Load Documents for url or pdf
def load_document(source):
  """ Check if the source is URL or Pdf Document 
      then, load document accordingly
  """
  docs = None

  # URL- if string starts with "http"
  if isinstance(source, str) and source.startswith("http"):
    # Pdf URL if endswith ".pdf"
    if source.lower().endswith(".pdf"):
      import requests
      response = requests.get(source)
      return get_pdf_contents(response.content)

      # or we can use OnlinePdfLoader from langchain
      # OnlinePdfLoader can handle only native, text-based Pdf urls
      # docs = OnlinePDFLoader(source).load()
      # return docs

    loader = WebBaseLoader(source)
    docs = loader.load()
    return docs
  
  # Uploaded PDF Document - if hasattr "read"
  if hasattr(source, "read"):
    return get_pdf_contents(source.read())

  return None
  

# Chunk text for storing meaningful embedding in chromadb
def chunk_text(docs):
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
  )

  chunks = splitter.split_documents(docs)
  return chunks


# Build knowledge base with chunked text
def build_knowledge_base(chunks):
  chroma_client = chromadb.Client()

  chroma_client.delete_collection(name="knowledge_base_ai_chatbot")
  collection = chroma_client.create_collection(name="knowledge_base_ai_chatbot")

  # Make list of documents, metadatas and ids to add to collection
  docs, metadatas, ids = [], [], []
  for i, doc in enumerate(chunks):
    docs.append(doc.page_content)
    ids.append(f"doc_{i}")

    # Append clean metadata, for chromadb to process
    clean_meta = {
      k: str(v) if v is not None else "unknown"
      for k,v in doc.metadata.items()
      if isinstance(v, (str, int, bool, float)) or v is None
    }
    metadatas.append(clean_meta)

  collection.add(
    documents=docs,
    ids=ids,
    metadatas=metadatas
  )

  logger.info("Embed Added to Collection")
  return collection


# Retrieve data from collection on basis of question
def retrieve(collection: chromadb.Collection, query, n_results=6, min_similarity=0.5):
  results = collection.query(query_texts=[query], n_results=n_results)

  filtered_docs = []
  filtered_meta = []

  for i in range(len(results["documents"][0])):
    distance = results["distances"][0][i]
    similarity = 1/(distance+1)

    if (similarity >= min_similarity):
      filtered_docs.append(results["documents"][0][i])
      filtered_meta.append(results["metadatas"][0][i])

  if not filtered_docs:
    filtered_docs.append(results["documents"][0][0])
    filtered_meta.append(results["metadatas"][0][0])

  # Make single string out of documents and metadata for AI to cite
  docs_string = "\n\n".join(f"Doc {i}: {doc}\n Meta {i}: {filtered_meta[i]}" for i,doc in enumerate(filtered_docs))

  return docs_string

@st.cache_resource
def get_llm():
  return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_output_tokens=500)

# Get llm stored in cache
llm = get_llm()

# Create the prompt and pipeline

prompt = ChatPromptTemplate.from_messages([
  ('system', """You are a helpful Q/A assistant. 
    Answer the user's question using ONLY the provided context.
    Cite the source metadata with your answer.
    If context doesn't contain the answer say "Sorry! No relatable data found.".
    Don't make anything up on your own. Keep responses to 3-4 sentences maximum.
    
    Format response as: 
      [Answer text]
      \nSource: citattions from document or page number from metadata if available or citation
   """),
  ('human', """Answer from the given context only.
    Context: {context}
    Question: {query}
    Answer:
   """
  )
])

chain = prompt | llm | StrOutputParser()


# Ask User Question return LLM Answer
def ask(collection, question):
  # Retrieve relevant data from collection
  context = retrieve(collection, question)

  # Get the llm answer
  answer = chain.invoke({
    "query": question,
    "context": context
  })

  return answer


# ====== PAGE CONFIG =======
st.title("KNOWLEDGE BASE CHATBOT")


# ====== SIDEBAR: Input Source ======
with st.sidebar:
  st.header("Add Knowledge Source")

  source = None
  source_type = st.radio("Choose input type: ", ["URL", "PDF Upload"])

  if source_type == "URL":
    url = st.text_input("Paste URL: ").strip()
    if url:
      source = url
  else:
    uploaded_file = st.file_uploader("Upload Pdf: ", type=["pdf"])
    if uploaded_file:
      source = uploaded_file

  clicked = st.button("Process")

if "collection" not in st.session_state:
  st.session_state.collection = None

if clicked and source:
  with st.spinner("Processing..."):
    st.session_state.messages = []
    st.session_state.collection = build_knowledge_base(chunk_text(load_document(source)))
  st.success("Document Loaded!")


# ====== MAIN: Chat Interface ======
if "messages" not in st.session_state:
  st.session_state.messages = []


# ======= DISPLAY PREVIOUS MESSAGES ======
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    st.write(msg["content"])


# ======= HANDLE NEW USER INPUT ======
if question := st.chat_input("Ask a question"):
  # Show user message
  with st.chat_message("user"):
    st.write(question)
  st.session_state.messages.append({"role": "user", "content": question})

  # Get Answer from AI
  try:
    answer = ask(st.session_state.collection, question.strip())
    with st.chat_message("assistant"):
      st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
  except Exception as e:
    logger.error(f"\nError Occured: {e}")
    st.error("Something Went Wrong!")
