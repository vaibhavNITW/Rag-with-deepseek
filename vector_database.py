from xml.dom.minidom import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#Step 1: upload & load raw PDF file(s)

pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path): 
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# THIS IS FOR TESTING PURPOSE

#file_path ='universal_decl.pdf'
#documents = load_pdf(file_path)
#print(len(documents))


#Step 2: Create Chunks

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
          chunk_overlap=200,
          add_start_index=True,)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# THIS IS FOR TESTING PURPOSE

#text_chunks = create_chunks(documents)
#print("chunks count: ", len(text_chunks))


#Step 3: Create Embeddings Model (use Deepseek R1 with ollama)
ollama_model_name = "deepseek-r1:1.5b"
def get_embeddings_model(ollama_model_name):
    embeddings = OllamaEmbeddings(model=ollama_model_name)
    return embeddings

#step 4: Create Vector Store (FAISS)
FAISS_DB_PATH = "vectorstore/db_faiss"
faiss_db=FAISS.from_documents(text_chunks, get_embeddings_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)