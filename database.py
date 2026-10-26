
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

path="data/Data Science_SQL.pdf" 

docs=PyPDFLoader(file_path=path)

documents=docs.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap =200,
)

chunks = splitter.split_documents(documents)

Embedding_model= HuggingFaceEmbeddings()

vectorestore=Chroma.from_documents(
    documents=chunks,
    embedding=Embedding_model,
    persist_directory="chroma-db"
    )


print("-----Succesfully Store VectoreDatabase----")