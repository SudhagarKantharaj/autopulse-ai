
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

manual_path = Path("data/diagnostic_manual.txt")
db_path = Path("chroma_db")

if not manual_path.exists():
    raise FileNotFoundError(
        "Manual not found. Check data/diagnostic_manual.txt"
    )

manual_text = manual_path.read_text(encoding="utf-8")

document = Document(
    page_content=manual_text,
    metadata={
        "source": str(manual_path),
        "document_type": "synthetic_engineering_manual",
    },
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
)
chunks = splitter.split_documents([document])

if db_path.exists():
    shutil.rmtree(db_path)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(db_path),
    collection_name="autopulse_manuals",
)

print(f"Created {len(chunks)} document chunks.")
print(f"Vector database saved to: {db_path.resolve()}")
print("Embeddings generated locally using Ollama.")

