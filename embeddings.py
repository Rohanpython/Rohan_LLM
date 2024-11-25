import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Load FAISS index from the local disk
def load_embeddings_offline(faiss_index_path="faiss_index"):
    try:
        vectorstore = FAISS.load_local(
            faiss_index_path,
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            ),
            allow_dangerous_deserialization=True
        )
        print("Embeddings loaded successfully from FAISS index.")
        return vectorstore
    except Exception as e:
        raise Exception(f"Error loading FAISS index: {e}")

# Extract text from PDFs with error handling
def get_pdf_text_with_metadata(docs):
    chunks = []
    metadata = []
    for pdf in docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    chunks.append(page_text)
                    metadata.append({
                        "source": os.path.basename(pdf),
                        "page": page_num + 1
                    })
            print(f"Successfully processed: {pdf}")
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
            continue
    return chunks, metadata

# Clustering text for topic separation and diversity
def cluster_texts(chunks, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    clustered_chunks = [[] for _ in range(num_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clustered_chunks[label].append(chunks[i])
    return clustered_chunks

# Split text into dense chunks with increased overlap and metadata
def get_chunks_with_metadata(text_list, metadata, cluster=False):
    raw_text = "\n".join(text_list)
    if not raw_text:
        raise ValueError("Raw text is empty, cannot create chunks.")
    
    # Split text into larger chunks with overlap for more context retention
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000, chunk_overlap=500, length_function=len)
    chunks = text_splitter.split_text(raw_text)

    # Optional clustering for richer embeddings by topic
    if cluster:
        clustered_chunks = cluster_texts(chunks)
        clustered_chunks_flattened = [chunk for cluster in clustered_chunks for chunk in cluster]
        
        # Adjust metadata to match the number of clustered chunks
        chunked_metadata = metadata * (len(clustered_chunks_flattened) // len(metadata) + 1)
        chunked_metadata = chunked_metadata[:len(clustered_chunks_flattened)]
        
        return clustered_chunks_flattened, chunked_metadata
    
    return chunks, metadata

# Create and save FAISS embeddings with metadata
def save_embeddings_to_faiss(chunks, chunked_metadata, faiss_index_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=chunked_metadata)
    vectorstore.save_local(faiss_index_path)
    print(f"Embeddings saved to {faiss_index_path}")

# Main function to process PDFs, create dense embeddings, and save FAISS index
def process_pdfs_in_folder_and_save_embeddings(folder_path, faiss_index_path, use_clustering=False):
    pdf_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    raw_text, metadata = get_pdf_text_with_metadata(pdf_files)
    
    if not raw_text:
        print(f"No text found in the PDFs within {folder_path}")
        return

    text_chunks, chunked_metadata = get_chunks_with_metadata(raw_text, metadata, cluster=use_clustering)
    
    if not text_chunks:
        print("No text chunks created.")
        return
    
    save_embeddings_to_faiss(text_chunks, chunked_metadata, faiss_index_path)
    vectorstore = load_embeddings_offline(faiss_index_path)
    print("All PDFs have been processed, embeddings have been saved, and FAISS index has been loaded.")

    return vectorstore

if __name__ == "__main__":
    folder_path = "data"
    faiss_index_path = "faiss_index"
    # Enable clustering to enrich embeddings
    vectorstore = process_pdfs_in_folder_and_save_embeddings(folder_path, faiss_index_path, use_clustering=True) 
