# ==================== main.py - FastAPI Server ====================
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # ← NEW: For streaming
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from collections import defaultdict
import statistics
from pathlib import Path
import json
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles

# Import our document intelligence system
from document_system import HybridDocumentSystem as DocumentIntelligenceSystem

# ==================== PERFORMANCE TRACKING ====================
class PerformanceTracker:
    """Track latency metrics for all queries"""
    
    def __init__(self):
        self.all_query_latencies = []   # All queries (including cache hits)
        self.cache_hit_latencies = []   # Only cache hits
        self.cache_miss_latencies = []  # Only cache misses (REAL latency)
        self.component_latencies = defaultdict(list)
        self.query_history = []
        
    def record_query(self, query_data: Dict[str, Any]):
        """Record a query with all timing breakdowns"""
        latency = query_data['total_latency']
        is_cache_hit = query_data.get('cache_hit', False)
        
        # Record in appropriate lists
        self.all_query_latencies.append(latency)
        
        if is_cache_hit:
            self.cache_hit_latencies.append(latency)
        else:
            self.cache_miss_latencies.append(latency)  # REAL search latency
        
        self.query_history.append(query_data)
        
        # Record component-wise latencies (only for cache misses)
        if not is_cache_hit:
            for component, component_latency in query_data.get('breakdown', {}).items():
                self.component_latencies[component].append(component_latency)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.all_query_latencies:
            return {
                'total_queries': 0,
                'error': 'No queries recorded'
            }
        
        # Calculate for ALL queries
        all_sorted = sorted(self.all_query_latencies)
        n_all = len(all_sorted)
        
        # Calculate for CACHE MISSES ONLY
        cache_miss_sorted = sorted(self.cache_miss_latencies) if self.cache_miss_latencies else [0]
        n_miss = len(cache_miss_sorted)
        
        # Calculate for CACHE HITS ONLY
        cache_hit_sorted = sorted(self.cache_hit_latencies) if self.cache_hit_latencies else [0]
        n_hit = len(cache_hit_sorted)
        
        return {
            'total_queries': n_all,
            'cache_hits': n_hit,
            'cache_misses': n_miss,
            'cache_hit_rate': round((n_hit / n_all * 100), 2) if n_all > 0 else 0,
            
            'all_queries': {
                'avg_latency': round(statistics.mean(self.all_query_latencies), 2),
                'median_latency': round(statistics.median(self.all_query_latencies), 2),
                'p95_latency': round(all_sorted[int(n_all * 0.95)], 2),
                'p99_latency': round(all_sorted[int(n_all * 0.99)] if n_all > 1 else all_sorted[-1], 2),
            },
            
            'cache_hits_only': {
                'avg_latency': round(statistics.mean(self.cache_hit_latencies), 2) if n_hit > 0 else 0,
                'p95_latency': round(cache_hit_sorted[int(n_hit * 0.95)], 2) if n_hit > 0 else 0,
            },
            
            'cache_misses_only': {
                'avg_latency': round(statistics.mean(self.cache_miss_latencies), 2) if n_miss > 0 else 0,
                'median_latency': round(statistics.median(self.cache_miss_latencies), 2) if n_miss > 0 else 0,
                'p50_latency': round(cache_miss_sorted[int(n_miss * 0.50)], 2) if n_miss > 0 else 0,
                'p95_latency': round(cache_miss_sorted[int(n_miss * 0.95)], 2) if n_miss > 0 else 0,
                'p99_latency': round(cache_miss_sorted[int(n_miss * 0.99)] if n_miss > 1 else cache_miss_sorted[-1], 2) if n_miss > 0 else 0,
                'min_latency': round(min(self.cache_miss_latencies), 2) if n_miss > 0 else 0,
                'max_latency': round(max(self.cache_miss_latencies), 2) if n_miss > 0 else 0,
                'std_dev': round(statistics.stdev(self.cache_miss_latencies), 2) if n_miss > 1 else 0,
            },
            
            'component_breakdown': {
                component: {
                    'avg': round(statistics.mean(latencies), 2),
                    'p95': round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0
                }
                for component, latencies in self.component_latencies.items()
            }
        }
    
    def reset(self):
        """Reset all metrics"""
        self.all_query_latencies = []
        self.cache_hit_latencies = []
        self.cache_miss_latencies = []
        self.component_latencies = defaultdict(list)
        self.query_history = []

# ==================== PYDANTIC MODELS ====================
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language query")
    k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    sector: Optional[str] = Field(None, description="Filter by sector (e.g., '128')")
    include_metadata: bool = Field(default=True, description="Include full metadata in response")
    
    # UPDATED CONFIG to avoid warnings
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What is the STP capacity?",
                "k": 5,
                "sector": "128",
                "include_metadata": True
            }
        }
    )

class SearchResult(BaseModel):
    rank: int
    content: str
    source: str
    page: int
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    question: str
    results: List[SearchResult]
    total_results: int
    latency_ms: float
    latency_breakdown: Dict[str, float]
    timestamp: str
    cache_hit: bool = False
    llm_answer: Optional[Dict[str, Any]] = None

class UploadRequest(BaseModel):
    pdf_path: str = Field(..., description="Path to PDF file to index")

class MetricsResponse(BaseModel):
    performance: Dict[str, Any]
    database_stats: Dict[str, Any]

# ==================== GLOBAL INSTANCES ====================
doc_system = DocumentIntelligenceSystem()
perf_tracker = PerformanceTracker()


# ==================== LIFESPAN ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    print("\n" + "="*70)
    print("🚀 SYSTEM STARTUP: Initializing Intelligence Engine...")
    print("="*70)
    
    # 1. Load Vector Store
    try:
        stats = doc_system.get_collection_stats()
        print(f"✓ Vector Store Loaded: {stats['total_chunks']} chunks")
        if 'has_bm25' in stats:
            print(f"✓ BM25 Index: {'Active' if stats['has_bm25'] else 'Not Found'}")
    except Exception as e:
        print(f"⚠ Vector Store Empty: {e}")

    # 2. WARM-UP ROUTINE
    print("\n🔥 Warming up Inference Engine...")
    start_warm = time.time()
    try:
        _ = doc_system.search_with_score("industry standard initialization", k=1)
        print(f"✓ Model Warm-up Complete: {(time.time() - start_warm):.4f}s")
    except Exception as e:
        print(f"⚠ Warm-up skipped: {e}")

    print(f"✓ System Ready to serve traffic!")
    print("="*70 + "\n")
    
    yield  # <--- Application runs here
    
    # --- SHUTDOWN PHASE ---
    print("\n" + "="*70)
    print("🛑 SYSTEM SHUTDOWN: Saving State...")
    metrics = perf_tracker.get_metrics()
    with open('performance_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Performance metrics saved")
    print("="*70)


# ==================== FASTAPI APP ====================
app = FastAPI(
    title="Real Estate Document Intelligence API",
    description="Query real estate documents using natural language",
    version="1.0.0",
    lifespan=lifespan
)

# ← ADD THIS:
# Serve static files
from pathlib import Path
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Serve the main HTML page
from fastapi.responses import FileResponse

@app.get("/ui")
async def serve_ui():
    """Serve the web interface"""
    return FileResponse(static_path / "index.html")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Real Estate Document Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query (POST)",
            "query_stream": "/query/stream (POST)",  # ← NEW
            "upload": "/upload (POST)",
            "metrics": "/metrics (GET)",
            "reset_metrics": "/metrics/reset (POST)",
            "docs": "/docs"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents using natural language (Cache-less version)
    """
    start_time = time.time()
    breakdown = {}
    
    try:
        # Build filter dict
        filter_dict = {"sector": request.sector} if request.sector else None
        
        # Vector/Hybrid search
        search_start = time.time()
        results = doc_system.search_with_score(
            query=request.question,
            k=request.k,
            filter_dict=filter_dict
        )
        breakdown['vector_search'] = (time.time() - search_start) * 1000
        
        # Format results
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append(SearchResult(
                rank=i + 1,
                content=doc.page_content,
                source=doc.metadata.get('source', 'Unknown'),
                page=doc.metadata.get('page_num', 0),
                score=round(float(score), 4),
                metadata=doc.metadata if request.include_metadata else None
            ))
        
        total_latency = (time.time() - start_time) * 1000
        
        # Record metrics for the benchmark report
        perf_tracker.record_query({
            'question': request.question,
            'total_latency': total_latency,
            'breakdown': breakdown,
            'cache_hit': False, # Always false now
            'timestamp': datetime.now().isoformat()
        })
        
        return QueryResponse(
            question=request.question,
            results=formatted_results,
            total_results=len(formatted_results),
            latency_ms=total_latency,
            latency_breakdown=breakdown,
            timestamp=datetime.now().isoformat(),
            cache_hit=False
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ==================== NEW: STREAMING ENDPOINT ====================
@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streaming query endpoint - returns search results + LLM answer as SSE stream
    """
    # Build filter
    filter_dict = None
    if request.sector:
        filter_dict = {"sector": request.sector}
    
    # Perform search
    search_start = time.time()
    results = doc_system.hybrid_search(
        query=request.question,
        k=request.k,
        filter_dict=filter_dict
    )
    search_latency = (time.time() - search_start) * 1000
    
    # Format results
    formatted_results = []
    for i, (doc, score) in enumerate(results):
        formatted_results.append({
            'rank': i + 1,
            'content': doc.page_content,
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page_num', 0),
            'score': round(float(score), 4),
            'metadata': doc.metadata if request.include_metadata else None
        })
    
    # Stream generator
    async def generate():
        # 1. Send search results immediately
        yield f"data: {json.dumps({'type': 'results', 'results': formatted_results, 'search_latency': search_latency})}\n\n"
        
        # 2. Stream LLM answer if enabled and results found
        if doc_system.use_llm and results:
            llm_result = doc_system.generate_answer_with_llm(
                request.question,
                results,
                max_chunks=3,
                stream=True
            )
            
            if 'stream' in llm_result:
                answer_text = ""
                for chunk in llm_result['stream']:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        answer_text += content
                        # Send each token
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                
                # Send completion
                yield f"data: {json.dumps({'type': 'done', 'answer': answer_text, 'sources': llm_result['sources']})}\n\n"
            elif 'error' in llm_result:
                yield f"data: {json.dumps({'type': 'error', 'error': llm_result['error']})}\n\n"
        else:
            # No LLM available
            yield f"data: {json.dumps({'type': 'done', 'answer': None})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(request: UploadRequest, background_tasks: BackgroundTasks):
    """Upload and index a new PDF document"""
    pdf_path = request.pdf_path
    
    if not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")
    
    background_tasks.add_task(index_document, pdf_path)
    
    return {
        "status": "processing",
        "message": f"Document {Path(pdf_path).name} is being indexed in background",
        "file": pdf_path
    }


async def index_document(pdf_path: str):
    """Background task to index a document"""
    try:
        doc_system.add_documents([pdf_path])
        print(f"✓ Successfully indexed: {pdf_path}")
    except Exception as e:
        print(f"✗ Failed to index {pdf_path}: {e}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics and database statistics"""
    try:
        performance_metrics = perf_tracker.get_metrics()
        db_stats = doc_system.get_collection_stats()
        
        return MetricsResponse(
            performance=performance_metrics,
            database_stats=db_stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset performance metrics"""
    perf_tracker.reset()
    query_cache.clear()
    
    return {
        "status": "success",
        "message": "Metrics and cache cleared"
    }


@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(query_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "cached_queries": list(query_cache.keys())
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear query cache"""
    query_cache.clear()
    return {
        "status": "success",
        "message": "Cache cleared"
    }


@app.get("/stats")
async def get_stats():
    """Get database statistics (for Streamlit sidebar)"""
    try:
        stats = doc_system.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ==================== RUN SERVER ====================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )