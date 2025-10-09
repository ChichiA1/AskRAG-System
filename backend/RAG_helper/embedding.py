from pathlib import Path
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go
import shutil
from backend.RAG_helper.doc_chunking import Chunker
from backend import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class VectorEmbedding:
    def __init__(self, encoder_model: str = config.ENCODER_MODEL):
        self.embedding = HuggingFaceEmbeddings(model_name=encoder_model)

    def create_vector(self):
        if config.db_folder.exists() and any(config.db_folder.iterdir()):
            shutil.rmtree(config.db_folder)  # Delete entire folder
            print(f"Deleted existing database folder")

        self.vectorstore = Chroma.from_documents(
            documents=Chunker().chunk(),
            embedding=self.embedding,
            persist_directory=str(config.db_folder)
        )
        print(f"Vectorstore created at {config.db_folder}")
        return self.vectorstore

    def load_vector(self):
        if not config.db_folder.exists() or not any(config.db_folder.iterdir()):
            raise FileNotFoundError(f"No vectorstore found at {config.db_folder}")

        print(f"Vectorstore loaded from {config.db_folder}")
        return Chroma(
            persist_directory=str(config.db_folder),
            embedding_function=self.embedding
        )

    def visual_rep(self):
        output = self.vectorstore._collection.get(include=["embeddings", "documents", "metadatas"])
        vectors = np.array(output["embeddings"])
        documents = output['documents']
        metadatas = output['metadatas']
        doc_types = [metadata['doc_type'] for metadata in metadatas]
        unique_doc_types = list({chunk.metadata.get('doc_type', 'unknown') for chunk in self.chunks})
        palette = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
        # Zip unique types to colors safely
        color_map = {t: palette[i % len(palette)] for i, t in enumerate(unique_doc_types)}
        # Map your actual doc_types to colors
        colours = [color_map.get(t, "gray") for t in doc_types]
        tsne = TSNE(n_components=3, random_state=42)
        reduced_vectors = tsne.fit_transform(vectors)
        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers',
            marker=dict(size=5, color=colours, opacity=0.8),
            text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
            hoverinfo='text'
        )])

        fig.update_layout(
            title='3D Chroma Vector Store Visualization',
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
            width=900,
            height=700,
            margin=dict(r=20, b=10, l=10, t=40)
        )
        fig.show()


if __name__ == "__main__":
    embedding = VectorEmbedding()
    embedding.create_vector()
    embedding.load_vector()
    embedding.visual_rep()
