import torch
import faiss
import numpy as np
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import GPT2Config

# === bGPT Imports ===
try:
    from lib.bgpt.utils import bGPTLMHeadModel 
    from lib.bgpt.config import PATCH_NUM_LAYERS, BYTE_NUM_LAYERS, HIDDEN_SIZE, PATCH_SIZE
except ImportError:
    print("Error: Could not import bGPT modules. Run this script from the project root.")
    exit(1)


# ==================== Configuration ====================
class RAGRetrieverConfig:
    """Configuration for bGPT Audio RAG"""
    # Model paths
    DEFAULT_MODEL_NAME = "./pretrained/bgpt/weights-audio.pth" 
    DEFAULT_PERSIST_PATH = "retriever_cache/bgpt_audio_storage"
    
    # Audio parameters
    SAMPLE_RATE = 8000
    CHUNK_DURATION = 1.0  
    CHUNK_OVERLAP = 0.5 
    
    # Retrieval parameters
    DEFAULT_TOP_K = 3


# ==================== bGPT Wrapper ====================
class BgptEmbeddingModel:
    """
    Wrapper to make bGPT act like an Embedding Model.
    It runs the audio through bGPT and extracts the mean hidden state.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        print(f"Loading bGPT for Embeddings from {checkpoint_path}...")
        
        # Configs
        patch_config = GPT2Config(
            num_hidden_layers=PATCH_NUM_LAYERS,
            max_length=512, 
            max_position_embeddings=512,
            hidden_size=HIDDEN_SIZE,
            n_head=HIDDEN_SIZE // 64,
            vocab_size=1,
        )
        byte_config = GPT2Config(
            num_hidden_layers=BYTE_NUM_LAYERS,
            max_length=PATCH_SIZE + 1,
            max_position_embeddings=PATCH_SIZE + 1,
            hidden_size=HIDDEN_SIZE,
            n_head=HIDDEN_SIZE // 64,
            vocab_size=256 + 1,
        )
        
        # Load Model
        self.model = bGPTLMHeadModel(patch_config, byte_config)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.to(device)
        self.model.eval()

    def get_sentence_embedding_dimension(self):
        return HIDDEN_SIZE

    def encode(self, inputs: List[List[int]], show_progress_bar=False):
        """
        :param inputs: List of raw byte lists [ [int, int...], [int, int...] ]
        """
        all_embeddings = []
        iterator = tqdm(inputs, desc="Embedding Audio") if show_progress_bar else inputs
        
        with torch.no_grad():
            for byte_seq in iterator:
                # Patch input
                patches = self._bytes_to_patches(byte_seq).to(self.device)
                
                # Forward Pass (Patch Transformer Only)
                patch_emb = self.model.patch_transformer(
                    inputs_embeds=self.model.linear_projection(patches.float().unsqueeze(-1))
                ).last_hidden_state
                
                # Mean Pooling (Average all patches to get 1 vector)
                embedding = torch.mean(patch_emb, dim=1).squeeze(0)
                
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                all_embeddings.append(embedding.cpu().numpy())

        return np.array(all_embeddings)

    def _bytes_to_patches(self, byte_list):
        """Converts flat byte list to [1, Num_Patches, Patch_Size]"""
        # Truncate if too long 
        max_len = 512 * PATCH_SIZE 
        if len(byte_list) > max_len:
            byte_list = byte_list[:max_len]

        # Pad to multiple of PATCH_SIZE
        remainder = len(byte_list) % PATCH_SIZE
        if remainder != 0:
            byte_list = byte_list + [0] * (PATCH_SIZE - remainder)
            
        # Reshape
        patches = torch.tensor(byte_list, dtype=torch.float32) 
        patches = patches.view(1, -1, PATCH_SIZE)
        return patches


# ==================== bGPT RAG Retriever ====================
class BgptRagRetriever:
    def __init__(self, model_path=None, persist_path=None):
        if model_path is None: model_path = RAGRetrieverConfig.DEFAULT_MODEL_NAME
        if persist_path is None: persist_path = RAGRetrieverConfig.DEFAULT_PERSIST_PATH
        
        self.persist_path = persist_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # bGPT wrapper
        self.embedding_model = BgptEmbeddingModel(model_path, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS
        self.index = None
        self.doc_store = {} 
        self.next_id = 0
        
        self._load_index()

    def index_audio_dataset(self, folder_path):
        """
        Reads all .wav files in folder, chunks them, and indexes using bGPT features.
        """
        import glob
        
        # .wav files
        wav_files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)
        print(f"Found {len(wav_files)} WAV files. Starting indexing...")
        
        all_embeddings = []
        ids = []
        chunk_batch = []
        
        # 8000 Hz * Duration
        chunk_size = int(RAGRetrieverConfig.SAMPLE_RATE * RAGRetrieverConfig.CHUNK_DURATION)
        stride = int(RAGRetrieverConfig.SAMPLE_RATE * (RAGRetrieverConfig.CHUNK_DURATION - RAGRetrieverConfig.CHUNK_OVERLAP))
        
        for fpath in tqdm(wav_files, desc="Processing Files"):
            try:
                with open(fpath, "rb") as f:
                    raw_data = list(f.read())
                    if len(raw_data) > 44:
                        raw_data = raw_data[44:]
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue
            
            # Create Chunks
            for start in range(0, len(raw_data), stride):
                end = start + chunk_size
                if end > len(raw_data): break
                
                chunk_bytes = raw_data[start:end]
                
                # Store in memory
                self.doc_store[self.next_id] = chunk_bytes
                chunk_batch.append(chunk_bytes)
                ids.append(self.next_id)
                self.next_id += 1
        
        # Encode all chunks in one go 
        if chunk_batch:
            print(f"Generating Embeddings for {len(chunk_batch)} chunks...")
            embeddings = self.embedding_model.encode(chunk_batch, show_progress_bar=True)
            
            if self.index is None:
                print("Creating new FAISS Index (Inner Product)...")
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            
            print("Adding to FAISS...")
            self.index.add_with_ids(embeddings.astype('float32'), np.array(ids).astype('int64'))
            self._save_index()
            print("Done.")

    def retrieve(self, query_bytes: List[int], k=3):
        """
        Retrieve context using raw audio bytes as query
        """
        if self.index is None or self.index.ntotal == 0:
            print("Index is empty!")
            return []

        # Encode Query
        query_emb = self.embedding_model.encode([query_bytes])[0]
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "id": int(idx),
                    "bytes": self.doc_store[idx],
                    "score": float(distances[0][i])
                })
                
        return results

    def _save_index(self):
        if not os.path.exists(self.persist_path): os.makedirs(self.persist_path)
        faiss.write_index(self.index, os.path.join(self.persist_path, "index.bin"))
        with open(os.path.join(self.persist_path, "docs.pkl"), "wb") as f:
            pickle.dump({"store": self.doc_store, "next_id": self.next_id}, f)

    def _load_index(self):
        index_file = os.path.join(self.persist_path, "index.bin")
        doc_file = os.path.join(self.persist_path, "docs.pkl")
        
        if os.path.exists(index_file) and os.path.exists(doc_file):
            print(f"Loading index from {self.persist_path}...")
            self.index = faiss.read_index(index_file)
            with open(doc_file, "rb") as f:
                data = pickle.load(f)
                self.doc_store = data["store"]
                self.next_id = data["next_id"]
            print(f"Loaded {self.index.ntotal} chunks.")
        else:
            print("No existing index found.")


# ==================== Main Workflow ====================
if __name__ == '__main__':
    # Setup
    AUDIO_DATASET_PATH = "datasets/LibriSpeech/wav" 
    
    retriever = BgptRagRetriever()

    # Index 
    if retriever.index is None or retriever.index.ntotal == 0:
        if os.path.exists(AUDIO_DATASET_PATH):
            retriever.index_audio_dataset(AUDIO_DATASET_PATH)
        else:
            print(f"Warning: Dataset path {AUDIO_DATASET_PATH} not found.")

    # Test Retrieval
    if retriever.index and retriever.index.ntotal > 0:
        print("\n--- Testing Retrieval ---")
        first_id = list(retriever.doc_store.keys())[0]
        query_chunk = retriever.doc_store[first_id]
        
        results = retriever.retrieve(query_chunk, k=3)
        
        for i, res in enumerate(results):
            print(f"Result {i+1}: ID {res['id']}, Score {res['score']:.4f}, Bytes {len(res['bytes'])}")