from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL = "llama3.2:latest"
# MODEL_INSTRUCT = "llama-3.2"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
db_folder = Path(__file__).resolve().parent / "vector_db"
doc_path = Path(__file__).resolve().parent / "utils" / "generated_docs"
llama_base_url = "http://localhost:11434/v1"
