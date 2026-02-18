# ==================== document_system.py - HYBRID SEARCH + LLM ====================
import os
from pathlib import Path
import pdfplumber
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from datetime import datetime
import json
import re
import pickle
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from sentence_transformers import CrossEncoder
from groq import Groq  
from llama_parse import LlamaParse

class HybridDocumentSystem:
    """Generic Document Intelligence System - Works for ANY PDF"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings
        print("🔧 Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.vectorstore = None
        
        # Text splitter - generic chunk size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # BM25 components
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_corpus = []

        # LOAD BM25 IMMEDIATELY
        if self._load_bm25_index():
            print("✓ BM25 index loaded successfully!")
        else:
            print("⚠️ BM25 index not found on disk (will be created on reindex)")

        # Cross-encoder reranker
        print("🔧 Initializing cross-encoder reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # ==================== NEW: Initialize LLM (Groq) ====================
        print("🤖 Initializing LLM (Groq)...")
        api_key = os.environ.get('GROQ_API_KEY')
        if api_key:
            self.llm_client = Groq(api_key=api_key)
            self.use_llm = True
            print("✓ LLM initialized (Groq Llama 3.3 70B)!")
        else:
            self.llm_client = None
            self.use_llm = False
            print("⚠️  No GROQ_API_KEY found - LLM disabled")

        # ==================== NEW: Initialize LlamaParse ====================
        print("🦙 Initializing LlamaParse...")
        llama_key = os.environ.get('LLAMA_CLOUD_API_KEY')
        if llama_key:
            self.llama_parser = LlamaParse(
                api_key=llama_key,
                result_type="markdown",  # Get clean markdown
                num_workers=4,  # Parallel processing
                verbose=True,
                language="en"
            )
            self.use_llamaparse = True
            print("✓ LlamaParse initialized!")
        else:
            self.llama_parser = None
            self.use_llamaparse = False
            print("⚠️  No LLAMA_CLOUD_API_KEY found - LlamaParse disabled, using PDFPlumber")

        print("✓ Generic system with LlamaParse + reranking + LLM initialized!")
    
    # ==================== GENERIC PDF EXTRACTION ====================
    # ==================== GENERIC PDF EXTRACTION ====================
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        print(f"\n📄 Extracting: {Path(pdf_path).name}")
        
        # Try LlamaParse first if available
        if self.use_llamaparse:
            print("🦙 Using LlamaParse for extraction...")
            try:
                extracted = self._extract_with_llamaparse(pdf_path)
                if self._is_good_quality(extracted['full_text']):
                    print("✅ LlamaParse extraction successful!")
                    return extracted
                else:
                    print("⚠️  LlamaParse gave poor results, falling back to PDFPlumber...")
            except Exception as e:
                print(f"⚠️  LlamaParse failed: {e}, falling back to PDFPlumber...")
        
        # Fallback to PDFPlumber
        print("📄 Using PDFPlumber for extraction...")
        extracted = self._extract_with_pdfplumber(pdf_path)
        return extracted
    
    def _extract_with_llamaparse(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text using LlamaParse (best for encoded PDFs)"""
        # Parse the PDF
        documents = self.llama_parser.load_data(pdf_path)
        
        # Combine all pages
        full_text = ""
        pages = []
        
        for i, doc in enumerate(documents, start=1):
            page_text = doc.text
            full_text += f"\n[PAGE {i}]\n{page_text}\n"
            pages.append({
                'page_num': i,
                'text': page_text
            })
        
        extracted = {
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'num_pages': len(pages),
            'full_text': full_text,
            'pages': pages,
            'extraction_method': 'LlamaParse'
        }
        
        print(f"✓ Extracted {len(full_text)} characters from {len(pages)} pages")
        return extracted
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Fallback: Extract using PDFPlumber"""
        extracted = {
            'file_name': Path(pdf_path).name,
            'file_path': pdf_path,
            'num_pages': 0,
            'full_text': '',
            'pages': [],
            'extraction_method': 'PDFPlumber'
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            extracted['num_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                
                tables = page.extract_tables()
                if tables:
                    for t_idx, table in enumerate(tables):
                        table_text = self._generic_table_to_text(table, page_num, t_idx)
                        page_text += f"\n\n{table_text}\n"
                
                extracted['pages'].append({
                    'page_num': page_num,
                    'text': page_text
                })
                
                extracted['full_text'] += f"\n[PAGE {page_num}]\n{page_text}\n"
        
        print(f"✓ Extracted {len(extracted['full_text'])} characters from {extracted['num_pages']} pages")
        return extracted
    
    def _is_good_quality(self, text: str) -> bool:
        """Check if extracted text is good quality"""
        if not text or len(text.strip()) < 100:
            return False
        
        # Check for CID encoding issues
        if "(cid:" in text.lower():
            print("⚠️  Detected CID encoding issues")
            return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 50:
            print(f"⚠️  Too few words extracted: {len(words)}")
            return False
        
        # Check for too many non-printable characters
        printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
        if printable_ratio < 0.7:
            print(f"⚠️  Too many non-printable chars: {printable_ratio:.2%}")
            return False
        
        return True
    
    def _generic_table_to_text(self, table_data: List[List[str]], page_num: int, table_idx: int) -> str:
        if not table_data or len(table_data) < 2:
            return ""
        
        cleaned_table = [
            [str(cell).replace('\n', ' ').strip() if cell is not None else "" 
             for cell in row]
            for row in table_data
        ]
        cleaned_table = [row for row in cleaned_table if any(cell for cell in row)]
        
        if len(cleaned_table) < 2:
            return ""
        
        headers = cleaned_table[0]
        lines = [f"\n=== TABLE {table_idx + 1} (Page {page_num}) ==="]
        
        for row in cleaned_table[1:]:
            if not any(cell for cell in row):
                continue
            
            phrases = []
            for header, value in zip(headers, row):
                if header and value:
                    phrases.append(f"{header} is {value}")
            
            if phrases:
                sentence = ", ".join(phrases) + "."
                lines.append(sentence)
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    # ==================== CHUNKING ====================
    def create_chunks(self, extracted_data: Dict[str, Any]) -> List[Document]:
        print(f"\n✂️  Creating chunks for {extracted_data['file_name']}...")
        
        texts = self.text_splitter.split_text(extracted_data['full_text'])
        
        documents = []
        for i, text in enumerate(texts):
            page_num = self._find_page_number(text, extracted_data['pages'])
            
            metadata = {
                'source': extracted_data['file_name'],
                'file_path': extracted_data['file_path'],
                'chunk_id': i,
                'page_num': page_num,
                'total_chunks': len(texts),
                'total_pages': extracted_data['num_pages']
            }
            
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        print(f"✓ Created {len(documents)} chunks")
        return documents
    
    def _find_page_number(self, text: str, pages: List[Dict]) -> int:
        for page in pages:
            if text[:100] in page['text']:
                return page['page_num']
        return 1
    
    # ==================== INDEXING ====================
    def add_documents(self, pdf_paths: List[str]):
        print("\n" + "="*70)
        print("📚 PROCESSING DOCUMENTS")
        print("="*70)
        
        all_documents = []
        
        for pdf_path in pdf_paths:
            extracted = self.extract_from_pdf(pdf_path)
            chunks = self.create_chunks(extracted)
            all_documents.extend(chunks)
        
        print(f"\n💾 Storing {len(all_documents)} chunks in ChromaDB...")
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=all_documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                client=self.chroma_client
            )
        else:
            self.vectorstore.add_documents(all_documents)
        
        print(f"✓ ChromaDB index created!")
        
        print(f"\n🔍 Building BM25 keyword index...")
        self._build_bm25_index(all_documents)
        
        print(f"✓ Successfully indexed all documents!")
        return len(all_documents)
    
    def _build_bm25_index(self, documents: List[Document]):
        self.bm25_documents = documents
        self.bm25_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        bm25_path = Path(self.persist_directory) / "bm25_index.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25_index,
                'documents': self.bm25_documents,
                'corpus': self.bm25_corpus
            }, f)
        
        print(f"✓ BM25 index saved")
    
    def _load_bm25_index(self):
        bm25_path = Path(self.persist_directory) / "bm25_index.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25_index = data['bm25']
                self.bm25_documents = data['documents']
                self.bm25_corpus = data['corpus']
            return True
        return False
    
    # ==================== HYBRID SEARCH ====================
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filter_dict: Dict[str, Any] = None
    ) -> List[Tuple[Document, float]]:
        
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        
        if self.bm25_index is None:
            if not self._load_bm25_index():
                print("⚠️  BM25 index not found, using vector-only search")
                return self.vectorstore.similarity_search_with_score(query, k=k)
        
        # 1. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        if bm25_scores.max() > 0:
            bm25_scores_norm = bm25_scores / bm25_scores.max()
        else:
            bm25_scores_norm = bm25_scores
        
        # 2. Vector Search
        k_candidates = min(k * 4, 50)
        vector_results = self.vectorstore.similarity_search_with_score(
            query, k=k_candidates, filter=filter_dict
        )
        
        vector_score_map = {}
        for doc, score in vector_results:
            similarity = 1 / (1 + score)
            vector_score_map[doc.page_content] = similarity
        
        # 3. Combine scores
        combined_results = []
        for i, doc in enumerate(self.bm25_documents):
            bm25_score = bm25_scores_norm[i]
            vector_score = vector_score_map.get(doc.page_content, 0)
            combined_score = (1 - alpha) * bm25_score + alpha * vector_score
            
            if filter_dict:
                match = all(doc.metadata.get(key) == value for key, value in filter_dict.items())
                if not match: continue
            
            combined_results.append((doc, combined_score))
        
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Rerank
        k_candidates = min(k * 4, 50)
        candidates = combined_results[:k_candidates]
        reranked_results = self.rerank_results(query, candidates, top_k=k)
        
        return reranked_results
    
    def rerank_results(self, query:str, candidates: List[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        if not candidates:
           return []
        
        pairs = [[query, doc.page_content] for doc, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)
    
        reranked = [
            (candidates[i][0], float(rerank_scores[i])) 
            for i in range(len(candidates))
        ]
    
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    # ==================== NEW: LLM ANSWER GENERATION ====================
    def generate_answer_with_llm(
        self,
        query: str,
        top_chunks: List[Tuple[Document, float]],
        max_chunks: int = 3,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer with LLM (supports streaming)
        
        Args:
            query: User's question
            top_chunks: Retrieved chunks with scores
            max_chunks: How many chunks to use (default: 3)
            stream: Enable streaming (default: False)
        
        Returns:
            Dict with 'answer' (or 'stream' if streaming), 'sources', etc.
        """
        if not self.use_llm:
            return {
                'answer': None,
                'error': 'LLM not configured',
                'sources': []
            }
        
        # Build context from top chunks
        chunks_to_use = top_chunks[:max_chunks]
        context_parts = []
        sources = []
        
        for i, (doc, score) in enumerate(chunks_to_use, 1):
            context_parts.append(f"[Source {i}]\n{doc.page_content}\n")
            sources.append({
                'rank': i,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page_num', 'Unknown'),
                'score': round(float(score), 4)
            })
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are analyzing environmental clearance documents. Extract the precise answer to the user's question from the provided sources.

Sources:
{context}

Question: {query}

Instructions:
1. Extract ONLY the specific value requested (with units)
2. Be concise - just the answer (e.g., "750 KLD", "422.93 crores", "Sector 128")
3. If the sources mention multiple values (Previous/Proposed/Total), use the TOTAL value
4. If not found in sources, say: "Information not found in provided sources"

Answer:"""
        
        try:
            if stream:
                # Return streaming response
                stream_response = self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                    stream=True
                )
                
                return {
                    'stream': stream_response,
                    'sources': sources,
                    'model': 'llama-3.3-70b-versatile'
                }
            else:
                # Regular non-streaming response
                response = self.llm_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150
                )
                
                answer = response.choices[0].message.content.strip()
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'model': 'llama-3.3-70b-versatile'
                }
        
        except Exception as e:
            return {
                'answer': None,
                'error': str(e),
                'sources': sources
            }
    
    # ==================== SEARCH INTERFACE ====================
    def search(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None):
        """Standard search interface"""
        results = self.hybrid_search(query, k=k, filter_dict=filter_dict)
        
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'content': doc.page_content,
                'metadata': doc.metadata,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page_num', 'Unknown'),
                'score': score
            })
        
        return formatted_results
    
    def search_with_score(self, query: str, k: int = 5, filter_dict: Dict[str, Any] = None):
        """Backward compatibility alias"""
        return self.hybrid_search(query, k=k, filter_dict=filter_dict)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        
        collection = self.chroma_client.get_collection(self.collection_name)
        count = collection.count()
        
        all_docs = self.vectorstore.get()
        unique_sources = set()
        
        if all_docs and 'metadatas' in all_docs:
            for metadata in all_docs['metadatas']:
                if 'source' in metadata:
                    unique_sources.add(metadata['source'])
        
        return {
            'total_chunks': count,
            'unique_documents': len(unique_sources),
            'documents': list(unique_sources),
            'has_bm25': self.bm25_index is not None
        }
    
    def reset_database(self):
        """Clear all data"""
        print("⚠️  Resetting database...")
        self.chroma_client.reset()
        self.vectorstore = None
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_corpus = []
        
        bm25_path = Path(self.persist_directory) / "bm25_index.pkl"
        if bm25_path.exists():
            bm25_path.unlink()
        
        print("✓ Database cleared!")

# Backward compatibility
DocumentIntelligenceSystem = HybridDocumentSystem