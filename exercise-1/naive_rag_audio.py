import torch
import numpy as np
from transformers import GPT2Config
from bgpt.utils import bGPTLMHeadModel 
from bgpt.config import PATCH_NUM_LAYERS, BYTE_NUM_LAYERS, HIDDEN_SIZE, PATCH_SIZE
import faiss
import numpy as np
import os
import pickle
import time
import gc
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import tqdm
class BgptEmbeddingModel:
    """
    Wrapper to make bGPT act like a SentenceTransformer for RAG.
    It extracts the mean hidden state of the audio patches.
    """
    def __init__(self, checkpoint_path, device="cpu"):
        self.device = device
        print(f"Loading bGPT for Embeddings from {checkpoint_path}...")
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
        
        # load model 
        self.model = bGPTLMHeadModel(patch_config, byte_config)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.to(device)
        self.model.eval()

    def get_sentence_embedding_dimension(self):
        return HIDDEN_SIZE

    def encode(self, inputs: List[List[int]], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        """
        :param inputs: List of raw byte lists [ [int, int...], [int, int...] ]
        """
        all_embeddings = []
        
        # We process one by one or in small batches to save RAM
        iterator = tqdm(inputs, desc="Embedding Audio") if show_progress_bar else inputs
        
        with torch.no_grad():
            for byte_seq in iterator:
                # 1. Prepare Input (Patchify)
                # We reuse the padding logic but simplified for just feature extraction
                # Helper to convert raw bytes -> patches tensor
                patches = self._bytes_to_patches(byte_seq).to(self.device)
                masks = torch.ones(patches.shape[:2], dtype=torch.long).to(self.device)

                # 2. Forward Pass (Get Hidden States)
                # bGPT returns (logits, hidden_states) if configured, or we access internal transformer
                # We need the output of the PATCH level transformer
                
                # Forward pass through patch_transformer directly to get high-level features
                # Note: bGPTLMHeadModel usually handles the hierarchy. 
                # We can run the full model and grab the last hidden state of the patch transformer.
                
                patch_emb = self.model.patch_transformer(
                    inputs_embeds=self.model.linear_projection(patches.float().unsqueeze(-1))
                ).last_hidden_state
                
                # patch_emb shape: [1, Num_Patches, Hidden_Size]
                
                # 3. Mean Pooling
                # Average all patches to get one vector for the whole audio chunk
                # Shape: [Hidden_Size]
                embedding = torch.mean(patch_emb, dim=1).squeeze(0)
                
                if normalize_embeddings:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                
                all_embeddings.append(embedding.cpu().numpy())

        if convert_to_numpy:
            return np.array(all_embeddings)
        return all_embeddings

    def _bytes_to_patches(self, byte_list):
        """Converts flat byte list to [1, Num_Patches, Patch_Size]"""
        # Pad to multiple of PATCH_SIZE
        remainder = len(byte_list) % PATCH_SIZE
        if remainder != 0:
            byte_list = byte_list + [0] * (PATCH_SIZE - remainder)
            
        # Reshape
        patches = torch.tensor(byte_list, dtype=torch.float32) # Float for projection
        patches = patches.view(1, -1, PATCH_SIZE)
        return patches

# ==================== Configuration ====================
class RAGRetrieverConfig:
    # Point to your bGPT checkpoint
    DEFAULT_MODEL_NAME = "./pretrained/bgpt/weights-audio.pth" 
    DEFAULT_PERSIST_PATH = "retriever_cache/bgpt_audio_storage"
    
    # Audio parameters
    SAMPLE_RATE = 8000
    CHUNK_DURATION = 1.0  # Seconds
    CHUNK_OVERLAP = 0.5   # Seconds
    
    # RAG parameters
    DEFAULT_TOP_K = 3

# ==================== bGPT RAG Retriever ====================
class BgptRagRetriever:
    def __init__(self, model_path=None, persist_path=None):
        if model_path is None: model_path = RAGRetrieverConfig.DEFAULT_MODEL_NAME
        if persist_path is None: persist_path = RAGRetrieverConfig.DEFAULT_PERSIST_PATH
        
        self.persist_path = persist_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- KEY CHANGE: Use custom bGPT wrapper ---
        self.embedding_model = BgptEmbeddingModel(model_path, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS
        self.index = None
        self.doc_store = {} # Maps ID -> Raw Audio Bytes
        self.next_id = 0
        
        self._load_index()

    def index_audio_dataset(self, folder_path):
        """
        Reads all .wav files in folder, chunks them, and indexes using bGPT features.
        """
        import glob
        import librosa
        
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        print(f"Indexing {len(wav_files)} files using bGPT features...")
        
        all_embeddings = []
        ids = []
        
        for fpath in tqdm(wav_files):
            # 1. Read Raw Bytes (assuming they are already 8k/8bit from your converter)
            # We read directly as bytes because bGPT expects discrete values 0-255
            with open(fpath, "rb") as f:
                raw_data = list(f.read())
                # Skip header if it exists (WAV header is usually 44 bytes)
                # Ideally, parse properly, but for raw bytes:
                if len(raw_data) > 44:
                    raw_data = raw_data[44:]
            
            # 2. Chunking (Byte-level sliding window)
            # 8000 Hz * 1 sec = 8000 bytes
            chunk_size = int(RAGRetrieverConfig.SAMPLE_RATE * RAGRetrieverConfig.CHUNK_DURATION)
            stride = int(RAGRetrieverConfig.SAMPLE_RATE * (RAGRetrieverConfig.CHUNK_DURATION - RAGRetrieverConfig.CHUNK_OVERLAP))
            
            for start in range(0, len(raw_data), stride):
                end = start + chunk_size
                if end > len(raw_data): break
                
                chunk_bytes = raw_data[start:end]
                
                # Store Data
                self.doc_store[self.next_id] = chunk_bytes
                
                # Prepare for Embedding
                # We collect them to batch encode or encode one by one
                # Here we encode one by one for simplicity
                emb = self.embedding_model.encode([chunk_bytes], show_progress_bar=False)[0]
                
                all_embeddings.append(emb)
                ids.append(self.next_id)
                self.next_id += 1
        
        # 3. Add to FAISS
        if len(all_embeddings) > 0:
            embeddings_matrix = np.array(all_embeddings).astype('float32')
            
            if self.index is None:
                # bGPT features are not normalized by default unless we force it
                # Using Inner Product (IP) usually requires normalization
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            
            self.index.add_with_ids(embeddings_matrix, np.array(ids).astype('int64'))
            self._save_index()
            print(f"Indexed {len(ids)} chunks.")

    def retrieve(self, query_bytes: List[int], k=3):
        """
        Retrieve context using raw audio bytes as query
        """
        # 1. Encode Query using bGPT
        query_emb = self.embedding_model.encode([query_bytes])[0]
        query_emb = query_emb.reshape(1, -1).astype('float32')
        
        # 2. Search
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.doc_store[idx])
                
        return results

    def _save_index(self):
        if not os.path.exists(self.persist_path): os.makedirs(self.persist_path)
        faiss.write_index(self.index, os.path.join(self.persist_path, "index.bin"))
        with open(os.path.join(self.persist_path, "docs.pkl"), "wb") as f:
            pickle.dump({"store": self.doc_store, "next_id": self.next_id}, f)

    def _load_index(self):
        if os.path.exists(os.path.join(self.persist_path, "index.bin")):
            self.index = faiss.read_index(os.path.join(self.persist_path, "index.bin"))
            with open(os.path.join(self.persist_path, "docs.pkl"), "rb") as f:
                data = pickle.load(f)
                self.doc_store = data["store"]
                self.next_id = data["next_id"]