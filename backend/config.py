from pathlib import Path

MODEL = "llama3:8b"
MODEL_INSTRUCT = "llama-3.2-3B-Instruct"
db_folder = "vector_db"
doc_path = Path(__file__).resolve().parent / "utils" / "generated_docs"
