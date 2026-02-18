# ==================== test_api.py - Performance Testing ====================
import requests
import time
import statistics
from typing import List, Dict, Any
import json
from datetime import datetime

class APIPerformanceTester:
    """Test API performance and generate metrics"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []


    def warm_up(self):
        """Fire dummy queries to wake up the system (NOT RECORDED)"""
        print("\n🔥 Warming up the engine (Ignoring these queries)...")
        try:
            # Fire 3 dummy requests to load models & DB into RAM
            dummy_questions = ["warmup query 1", "warmup query 2", "warmup query 3"]
            for q in dummy_questions:
                self.session.post(
                    f"{self.base_url}/query",
                    json={"question": q, "k": 1},
                    timeout=10
                )
                print(".", end="", flush=True)
            print(" Done! System is hot.\n")
        except Exception as e:
            print(f"Warm-up failed: {e}")
    
    def test_single_query(self, question: str, k: int = 5, sector: str = None) -> Dict[str, Any]:
        """Test a single query and measure latency"""
        start_time = time.time()
        
        payload = {
            "question": question,
            "k": k,
            "include_metadata": True
        }
        
        if sector:
            payload["sector"] = sector
        
        try:
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            
            client_latency = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'question': question,
                    'status': 'success',
                    'client_latency': client_latency,
                    'server_latency': data.get('latency_ms', 0),
                    'num_results': data.get('total_results', 0),
                    'cache_hit': data.get('cache_hit', False),
                    'breakdown': data.get('latency_breakdown', {})
                }
            else:
                return {
                    'question': question,
                    'status': 'error',
                    'error': response.text,
                    'client_latency': client_latency
                }
        
        except Exception as e:
            return {
                'question': question,
                'status': 'error',
                'error': str(e),
                'client_latency': (time.time() - start_time) * 1000
            }
    
    def run_test_suite(self, test_questions: List[Dict[str, Any]], iterations: int = 1):
        """Run all test questions"""
        self.warm_up()
        print("="*70)
        print("🧪 RUNNING PERFORMANCE TEST SUITE")
        print("="*70)
        print(f"Test questions: {len(test_questions)}")
        print(f"Iterations per question: {iterations}")
        print(f"Total queries: {len(test_questions) * iterations}")
        print("="*70)
        
        all_results = []
        
        for iteration in range(iterations):
            print(f"\n{'─'*70}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'─'*70}")
            
            for i, test_case in enumerate(test_questions, 1):
                question = test_case['question']
                k = test_case.get('k', 5)
                sector = test_case.get('sector')
                
                print(f"\n[{i}/{len(test_questions)}] Testing: {question[:50]}...")
                
                result = self.test_single_query(question, k, sector)
                all_results.append(result)
                
                if result['status'] == 'success':
                    print(f"  ✓ Latency: {result['server_latency']:.2f}ms")
                    print(f"  ✓ Results: {result['num_results']}")
                    print(f"  ✓ Cache: {'HIT' if result['cache_hit'] else 'MISS'}")
                else:
                    print(f"  ✗ Error: {result.get('error', 'Unknown')}")
                
                # Small delay between requests
                time.sleep(0.1)
        
        self.test_results = all_results
        return self.analyze_results()
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze test results and calculate metrics"""
        successful = [r for r in self.test_results if r['status'] == 'success']
        
        if not successful:
            return {
                'error': 'No successful queries',
                'total_queries': len(self.test_results)
            }
        
        latencies = [r['server_latency'] for r in successful]
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        cache_hits = sum(1 for r in successful if r.get('cache_hit', False))
        
        metrics = {
            'total_queries': len(self.test_results),
            'successful_queries': len(successful),
            'failed_queries': len(self.test_results) - len(successful),
            'cache_hits': cache_hits,
            'cache_hit_rate': round(cache_hits / len(successful) * 100, 2),
            
            # Latency metrics
            'avg_latency': round(statistics.mean(latencies), 2),
            'median_latency': round(statistics.median(latencies), 2),
            'min_latency': round(min(latencies), 2),
            'max_latency': round(max(latencies), 2),
            
            # Percentiles
            'p50_latency': round(sorted_latencies[int(n * 0.50)], 2),
            'p75_latency': round(sorted_latencies[int(n * 0.75)], 2),
            'p90_latency': round(sorted_latencies[int(n * 0.90)], 2),
            'p95_latency': round(sorted_latencies[int(n * 0.95)], 2),
            'p99_latency': round(sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1], 2),
            
            # Standard deviation
            'std_dev': round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2)
        }
        
        return metrics
    
    def print_results(self, metrics: Dict[str, Any]):
      
      """Pretty print test results"""
      print("\n" + "="*70)
      print("📊 PERFORMANCE TEST RESULTS")
      print("="*70)
    
      print(f"\n✓ Total Queries: {metrics['total_queries']}")
      print(f"✓ Successful: {metrics['successful_queries']}")
      print(f"✓ Failed: {metrics['failed_queries']}")
      print(f"✓ Cache Hit Rate: {metrics['cache_hit_rate']}%")
    
    # Separate cache hit vs miss latencies
      cache_miss_latencies = [r['server_latency'] for r in self.test_results 
                           if r['status'] == 'success' and not r.get('cache_hit', False)]
    
      if cache_miss_latencies:
        sorted_miss = sorted(cache_miss_latencies)
        n_miss = len(sorted_miss)
        
        print(f"\n{'─'*70}")
        print("CACHE MISS LATENCY (Real Search Performance) ⭐")
        print(f"{'─'*70}")
        print(f"Count:       {n_miss} queries")
        print(f"Average:     {statistics.mean(cache_miss_latencies):>8.2f} ms")
        print(f"Median:      {statistics.median(cache_miss_latencies):>8.2f} ms")
        print(f"Min:         {min(cache_miss_latencies):>8.2f} ms")
        print(f"Max:         {max(cache_miss_latencies):>8.2f} ms")
        print(f"\nPercentiles:")
        print(f"P95:         {sorted_miss[int(n_miss * 0.95)]:>8.2f} ms  ⭐⭐⭐")
        print(f"P99:         {sorted_miss[int(n_miss * 0.99)] if n_miss > 1 else sorted_miss[-1]:>8.2f} ms")
    
      print("="*70)
    
    def save_results(self, filename: str = 'test_results.json'):
        """Save results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.analyze_results(),
            'detailed_results': self.test_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")


# ==================== TEST QUESTIONS (18-20 as required) ====================
TEST_QUESTIONS = [
    # Basic Project Info
    {"question": "Who are the architects for the 222 Rajpur project?", "k": 3},
    {"question": "How many total residences are there in the 222 Rajpur community?", "k": 3},
    {"question": "What is the RERA Registration Number for 222 Rajpur?", "k": 3},
    {"question": "How many Townhouse units are available?", "k": 3},
    {"question": "Which forest does the project abut?", "k": 3},

    # Location & Connectivity
    {"question": "What is the distance to Max Super Speciality Hospital?", "k": 3},
    {"question": "How far is the Jolly Grant Airport from the project?", "k": 3},

    # Areas & Dimensions
    {"question": "What is the Built-up Area of a Townhouse unit?", "k": 3},
    {"question": "What is the Carpet Area of a Courtyard Villa?", "k": 3},
    {"question": "What is the plot size range for Forest Villas?", "k": 3},
    {"question": "What is the ceiling height provided in the residences?", "k": 3},

    # Specifications & Amenities
    {"question": "What brand of sanitary fixtures is provided in the bathrooms?", "k": 3},
    {"question": "What type of windows are installed in the residences?", "k": 3},
    {"question": "What flooring is used in the bedrooms?", "k": 3},
    {"question": "How many car parking spaces are provided per residence?", "k": 3},
    {"question": "What brand of microwave and oven is provided in the kitchen?", "k": 3},
    {"question": "What security systems are available in the community?", "k": 3},

    # Financial
    {"question": "What is the booking amount percentage in the Down Payment Plan?", "k": 3}
]


# ==================== RUN TESTS ====================
if __name__ == "__main__":
    tester = APIPerformanceTester()
    
    # Clear cache first for fair testing
    print("🧹 Clearing cache...")
    requests.post("http://localhost:8000/cache/clear")
    requests.post("http://localhost:8000/metrics/reset")
    
    # Run tests (3 iterations to test caching)
    metrics = tester.run_test_suite(TEST_QUESTIONS, iterations=3)
    
    # Print results
    tester.print_results(metrics)
    
    # Save to file
    tester.save_results('baseline_performance.json')