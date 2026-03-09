# ==================== caching_system.py ====================
import hashlib
import json
import pickle
from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
from datetime import datetime
import numpy as np

class LRUCache:
    """Least Recently Used Cache"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            self.cache[key] = value
            
            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total
        }


class EmbeddingCache:
    """Cache for query embeddings"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = LRUCache(max_size)
    
    def _make_key(self, text: str) -> str:
        """Create cache key from text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        key = self._make_key(text)
        cached = self.cache.get(key)
        
        if cached is not None:
            return np.array(cached)
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        key = self._make_key(text)
        # Store as list for JSON compatibility
        self.cache.set(key, embedding.tolist())
    
    def get_stats(self):
        return self.cache.get_stats()


class SearchResultsCache:
    """Cache for search results"""
    
    def __init__(self, max_size: int = 5000):
        self.cache = LRUCache(max_size)
    
    def _make_key(self, query: str, k: int, filter_dict: Optional[Dict] = None) -> str:
        """Create cache key"""
        key_data = {
            'query': query,
            'k': k,
            'filter': filter_dict or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, k: int, filter_dict: Optional[Dict] = None) -> Optional[List]:
        """Get cached results"""
        key = self._make_key(query, k, filter_dict)
        return self.cache.get(key)
    
    def set(self, query: str, k: int, results: List, filter_dict: Optional[Dict] = None):
        """Cache results"""
        key = self._make_key(query, k, filter_dict)
        
        # Serialize results (convert Document objects to dicts)
        serialized = []
        for doc, score in results:
            serialized.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            })
        
        self.cache.set(key, serialized)
    
    def get_stats(self):
        return self.cache.get_stats()


class LLMAnswerCache:
    """Cache for LLM-generated answers"""
    
    def __init__(self, max_size: int = 2000):
        self.cache = LRUCache(max_size)
    
    def _make_key(self, query: str, context_hash: str) -> str:
        """Create cache key from query and context"""
        key_str = f"{query}:{context_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _hash_context(self, chunks: List[Tuple]) -> str:
        """Create hash of context chunks"""
        # Use content of top 3 chunks
        content = ""
        for doc, _ in chunks[:3]:
            content += doc.page_content[:100]
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, chunks: List[Tuple]) -> Optional[Dict]:
        """Get cached answer"""
        context_hash = self._hash_context(chunks)
        key = self._make_key(query, context_hash)
        return self.cache.get(key)
    
    def set(self, query: str, chunks: List[Tuple], answer: Dict):
        """Cache answer"""
        context_hash = self._hash_context(chunks)
        key = self._make_key(query, context_hash)
        self.cache.set(key, answer)
    
    def get_stats(self):
        return self.cache.get_stats()


class CachingSystem:
    """Unified caching system"""
    
    def __init__(
        self,
        embedding_cache_size: int = 10000,
        results_cache_size: int = 5000,
        answer_cache_size: int = 2000
    ):
        self.embedding_cache = EmbeddingCache(embedding_cache_size)
        self.results_cache = SearchResultsCache(results_cache_size)
        self.answer_cache = LLMAnswerCache(answer_cache_size)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            'embedding_cache': self.embedding_cache.get_stats(),
            'results_cache': self.results_cache.get_stats(),
            'answer_cache': self.answer_cache.get_stats()
        }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_all_stats()
        
        print("\n💾 CACHE STATISTICS:")
        print("="*70)
        
        for cache_name, cache_stats in stats.items():
            print(f"\n{cache_name.upper().replace('_', ' ')}:")
            print(f"  Size: {cache_stats['size']}/{cache_stats['max_size']}")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.2f}%")
        
        print("\n" + "="*70)
    
    def clear_all(self):
        """Clear all caches"""
        self.embedding_cache.cache.clear()
        self.results_cache.cache.clear()
        self.answer_cache.cache.clear()