# ==================== benchmark_latency.py ====================
from document_system import HybridDocumentSystem
import time
import numpy as np
from typing import List, Dict

class LatencyBenchmark:
    """Benchmark latency with different configurations"""
    
    def __init__(self):
        self.system = HybridDocumentSystem()
        
        # Test queries
        self.test_queries = [
            "For 222 Rajpur, Dehradun, how many total residences are planned and over how many acres is the project spread?",
            "At 222 Rajpur, what types of residences are available?",
            "Is 222 Rajpur adjacent to any forest area? If yes, which one?",
            "What are the views offered from residences at 222 Rajpur, Dehradun?",
            "How far is Jolly Grant Airport from 222 Rajpur, Dehradun?",
            "What is the distance between 222 Rajpur and The Doon School?",
            "How long does it take to reach Pacific Mall from 222 Rajpur?",
            "How many Townhouse units are available at 222 Rajpur?",
            "Does the Townhouse at 222 Rajpur include a sky court?",
            "What is the plot size range for Courtyard Villas at 222 Rajpur?"
        ]
    
    def benchmark_configuration(
        self,
        name: str,
        use_reranking: bool = True,
        use_cache: bool = True,
        k: int = 5,
        iterations: int = 3
    ) -> Dict:
        """Benchmark a specific configuration"""
        
        print(f"\n{'='*70}")
        print(f"🔬 BENCHMARKING: {name}")
        print(f"{'='*70}")
        print(f"Reranking: {use_reranking} | Cache: {use_cache} | k={k}")
        
        # Configure system
        self.system.use_reranking = use_reranking
        
        # Clear cache if not using it
        if not use_cache:
            self.system.cache.clear_all()
        
        # Reset profiler
        self.system.profiler.reset()
        
        latencies = []
        
        # Run benchmark
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}:")
            
            for i, query in enumerate(self.test_queries, 1):
                start = time.time()
                
                results = self.system.hybrid_search(
                    query, 
                    k=k,
                    use_cache=use_cache
                )

                
                
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                
                # Print the latency, question, AND the LLM's answer
                print(f"  [{i:2d}/10] {latency:6.1f}ms - Q: {query}")
                print(f"      {'-'*50}")
        
        # Calculate statistics
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        
        stats = {
            'name': name,
            'config': {
                'reranking': use_reranking,
                'cache': use_cache,
                'k': k
            },
            'latencies': {
                'avg': np.mean(latencies),
                'median': np.median(latencies),
                'p50': latencies_sorted[int(n * 0.50)],
                'p95': latencies_sorted[int(n * 0.95)],
                'p99': latencies_sorted[min(int(n * 0.99), n-1)],
                'min': min(latencies),
                'max': max(latencies),
                'std': np.std(latencies)
            }
        }
        
        # Print summary
        print(f"\n📊 RESULTS:")
        print(f"  Average:  {stats['latencies']['avg']:.1f}ms")
        print(f"  Median:   {stats['latencies']['median']:.1f}ms")
        print(f"  P95:      {stats['latencies']['p95']:.1f}ms")
        print(f"  P99:      {stats['latencies']['p99']:.1f}ms")
        print(f"  Range:    {stats['latencies']['min']:.1f} - {stats['latencies']['max']:.1f}ms")
        
        # Get stage breakdown
        breakdown = self.system.profiler.get_breakdown()
        print(f"\n⚡ STAGE BREAKDOWN:")
        total = sum(breakdown.values())
        for stage, ms in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = (ms / total * 100) if total > 0 else 0
            print(f"  {stage:.<30} {ms:>6.1f}ms ({pct:>5.1f}%)")
        
        return stats
    
    def run_full_benchmark(self):
        """Run complete benchmark comparing all configurations"""
        
        print("\n" + "="*70)
        print("🚀 COMPREHENSIVE LATENCY BENCHMARK")
        print("="*70)
        
        configurations = [
            ("Baseline (Rerank + Cache)", True, True, 5),
            ("No Cache", True, False, 5),
            ("No Reranking", False, True, 5),
            ("No Rerank + No Cache", False, False, 5),
            ("Optimized (k=3)", True, True, 3),
        ]
        
        results = []
        
        for name, rerank, cache, k in configurations:
            stats = self.benchmark_configuration(
                name=name,
                use_reranking=rerank,
                use_cache=cache,
                k=k,
                iterations=3
            )
            results.append(stats)
            
            # Small delay between configs
            time.sleep(1)
        
        # Print comparison
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: List[Dict]):
        """Print comparison table"""
        
        print("\n" + "="*70)
        print("📊 CONFIGURATION COMPARISON")
        print("="*70)
        
        # Header
        print(f"\n{'Configuration':<30} {'Avg':>8} {'P95':>8} {'Improvement':>12}")
        print("-"*70)
        
        baseline_avg = results[0]['latencies']['avg']
        
        for stats in results:
            name = stats['name']
            avg = stats['latencies']['avg']
            p95 = stats['latencies']['p95']
            
            improvement = ((baseline_avg - avg) / baseline_avg * 100)
            improvement_str = f"{improvement:+.1f}%" if stats != results[0] else "baseline"
            
            print(f"{name:<30} {avg:>7.1f}ms {p95:>7.1f}ms {improvement_str:>12}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    benchmark = LatencyBenchmark()
    results = benchmark.run_full_benchmark()
    
    print("\n✅ Benchmark complete!")
    print("\nRecommendation:")
    
    # Find best config
    best = min(results, key=lambda x: x['latencies']['p95'])
    print(f"  Best configuration: {best['name']}")
    print(f"  P95 latency: {best['latencies']['p95']:.1f}ms")
    print(f"  Settings: Reranking={best['config']['reranking']}, "
          f"Cache={best['config']['cache']}, k={best['config']['k']}")