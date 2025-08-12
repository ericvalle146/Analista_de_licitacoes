git add .from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader as PDFLoader
from langchain_community.vectorstores import Chroma
import csv
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import pandas as pd


embedding = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="")


def ensure_chroma(path_pdf: str, persist_dir: str, tokens_size: int, tokens_overlap: int,modelo="gpt-4o-mini"):
    
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    
    else:
        
        loader = PDFLoader(path_pdf)
        docs = loader.load()
        texto_completo = ""
        for doc in docs:
            texto_completo += doc.page_content + "\n\n"
        spliter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=modelo,
            chunk_size=tokens_size,
            chunk_overlap=tokens_overlap
        )
        chunks = spliter.split_text(texto_completo)
        
        return Chroma.from_texts(
            texts=chunks,
            embedding=embedding,
            persist_directory=persist_dir
        )
    
def ensure_chroma_csv(path_csv="requisitos.csv", persist_dir="sata/Banco_db_req"):
    if os.path.exists(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
        result = db.get()
        ids = result["ids"]
        texts = result["documents"]  # já são strings
        db._map_by_id = dict(zip(ids, texts))
        return db

    # cria o banco se não existir
    df = pd.read_csv(path_csv, header=None, names=["id", "text"], dtype=str)
    ids = df["id"].tolist()
    texts = df["text"].tolist()
    metadatas = [{"numero": i} for i in ids]

    db = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        ids=ids,
        metadatas=metadatas,
        persist_directory=persist_dir
    )
    db.persist()
    db._map_by_id = dict(zip(ids, texts))
    return db