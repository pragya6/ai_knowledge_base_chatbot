# AI Knowledge Base Chatbot

import os
import logging
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, OnlinePDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv() # local path to api key

# For Streamlit Cloud
# if "GOOGLE_API_KEY" not in os.environ:
#     if "GOOGLE_API_KEY" in st.secrets:
#         os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"] 

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


# Embedding model from Huggingface- download once
@st.cache_resource
def get_embeddings():
  return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()


# Build knowledge base with chunked text
def build_knowledge_base(chunks):
  vectorstore = FAISS.from_documents(chunks, embeddings)
  return vectorstore


# Retrieve data from collection on basis of question
def retrieve(vectorstore, query, n_results=4, min_similarity=0.5):
  docs = vectorstore.similarity_search_with_score(query, k=n_results)

  filtered = []

  for doc, score in docs:
    similarity = 1/(score+1)

    if (similarity >= min_similarity):
      filtered.append(doc)

  if not filtered:
    filtered.append(docs[0][0])

  # Make a string of documents and metadata for AI to cite
  docs_string = "\n\n".join(f"Doc {i}: {doc.page_content}\n Meta {i}: {doc.metadata}" for i,doc in enumerate(filtered))

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
      \nSource: [source from metadata]
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
def ask(vectorsotre, question):
  # Retrieve relevant data from vectorsotre
  context = retrieve(vectorsotre, question)

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

if "vectorstore" not in st.session_state:
  st.session_state.vectorstore = None

if clicked and source:
  with st.spinner("Processing..."):
    st.session_state.messages = []
    st.session_state.vectorstore = build_knowledge_base(chunk_text(load_document(source)))
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
    answer = ask(st.session_state.vectorstore, question.strip())
    with st.chat_message("assistant"):
      st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
  except Exception as e:
    logger.error(f"\nError Occured: {e}")
    st.error("Something Went Wrong!")
