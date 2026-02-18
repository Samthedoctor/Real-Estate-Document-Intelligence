# ==================== accuracy_evaluator.py ====================
from pydoc import text
import requests
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime
import re

class AccuracyEvaluator:
    """Evaluate retrieval accuracy with ground truth"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.evaluation_results = []
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase, remove extra spaces
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def check_answer_in_text(self, text: str, expected_answer: str) -> bool:
        """Check if expected answer is in the text - GENERIC version"""
        text_lower = text.lower()
        
        # Extract just the numeric/core value from expected answer
        # e.g., "750 KLD" -> check for "750"
        # e.g., "EC25B3813HR5829482N" -> check for "EC25B3813HR5829482N"
        
        # Split expected answer into tokens
        tokens = expected_answer.lower().split()
        
        # Check if ANY significant token (not units) is present
        for token in tokens:
            # Skip common units
            if token in ['kld', 'sqm', 'sq.m', 'm', 'km', 'ecs', 'crores', 'tpd', 'kw', 'kwp', 'kva']:
                continue
            
            # Check if this token exists in text
            if token in text_lower:
                return True
        
        # Also check full expected answer
        if expected_answer.lower() in text_lower:
            return True
        
        return False
    
    def evaluate_single_query(
        self,
        question: str,
        expected_answer: str,
        keywords: List[str] = None,
        expected_source: str = None,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate a single query
        
        Args:
            question: The query
            expected_answer: What the answer should be
            keywords: Alternative keywords to match (for flexibility)
            expected_source: Which PDF should contain the answer
            k: Number of results to retrieve
        """
        # Query the API
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"question": question, "k": k, "include_metadata": True},
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    'question': question,
                    'status': 'api_error',
                    'top_1': False,
                    'top_3': False,
                    'top_5': False
                }
            
            data = response.json()
            results = data.get('results', [])
            
            if not results:
                return {
                    'question': question,
                    'status': 'no_results',
                    'top_1': False,
                    'top_3': False,
                    'top_5': False
                }
            
            # Check each result
            found_at_rank = None
            matching_result = None
            
            for i, result in enumerate(results[:k], 1):
                content = result.get('content', '')
                source = result.get('source', '')
                
                # Check if answer is in this result
                is_match = self.check_answer_in_text(content, expected_answer)
                
                # Also check source if specified
                source_match = True
                if expected_source:
                    source_match = expected_source.lower() in source.lower()
                
                if is_match and source_match:
                    found_at_rank = i
                    matching_result = result
                    break
            
            # Calculate accuracy metrics
            top_1 = found_at_rank == 1
            top_3 = found_at_rank is not None and found_at_rank <= 3
            top_5 = found_at_rank is not None and found_at_rank <= 5
            
            return {
                'question': question,
                'expected_answer': expected_answer,
                'status': 'evaluated',
                'found_at_rank': found_at_rank,
                'top_1': top_1,
                'top_3': top_3,
                'top_5': top_5,
                'matching_content': matching_result.get('content', '')[:200] if matching_result else None,
                'retrieved_results': len(results)
            }
            
        except Exception as e:
            return {
                'question': question,
                'status': 'error',
                'error': str(e),
                'top_1': False,
                'top_3': False,
                'top_5': False
            }
    
    def run_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run full evaluation suite"""
        print("="*70)
        print("🎯 RUNNING ACCURACY EVALUATION")
        print("="*70)
        print(f"Total test cases: {len(test_cases)}")
        print("="*70)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            question = test_case['question']
            expected = test_case['expected_answer']
            keywords = test_case.get('keywords')
            source = test_case.get('expected_source')
            
            print(f"\n[{i}/{len(test_cases)}] {question}")
            print(f"  Expected: {expected[:80]}...")
            
            result = self.evaluate_single_query(
                question=question,
                expected_answer=expected,
                keywords=keywords,
                expected_source=source,
                k=5
            )
            
            results.append(result)
            
            # Print result
            if result['status'] == 'evaluated':
                if result['top_1']:
                    print(f"  ✅ Found at Rank 1 (Top-1 ✓, Top-3 ✓)")
                elif result['top_3']:
                    print(f"  ✅ Found at Rank {result['found_at_rank']} (Top-3 ✓)")
                elif result['top_5']:
                    print(f"  ⚠️  Found at Rank {result['found_at_rank']} (Top-5 only)")
                else:
                    print(f"  ❌ NOT FOUND in top 5")
            else:
                print(f"  ❌ Error: {result.get('error', result['status'])}")
        
        self.evaluation_results = results
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate overall accuracy metrics"""
        evaluated = [r for r in self.evaluation_results if r['status'] == 'evaluated']
        
        if not evaluated:
            return {'error': 'No successful evaluations'}
        
        total = len(evaluated)
        top_1_correct = sum(1 for r in evaluated if r['top_1'])
        top_3_correct = sum(1 for r in evaluated if r['top_3'])
        top_5_correct = sum(1 for r in evaluated if r['top_5'])
        
        metrics = {
            'total_questions': total,
            'top_1_accuracy': round(top_1_correct / total * 100, 2),
            'top_3_accuracy': round(top_3_correct / total * 100, 2),
            'top_5_accuracy': round(top_5_correct / total * 100, 2),
            'top_1_correct': top_1_correct,
            'top_3_correct': top_3_correct,
            'top_5_correct': top_5_correct,
            
            # Breakdown by rank
            'rank_distribution': self._get_rank_distribution(evaluated)
        }
        
        return metrics
    
    def _get_rank_distribution(self, results: List[Dict]) -> Dict[int, int]:
        """Get distribution of ranks where answers were found"""
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 'not_found': 0}
        
        for r in results:
            rank = r.get('found_at_rank')
            if rank and rank <= 5:
                distribution[rank] += 1
            else:
                distribution['not_found'] += 1
        
        return distribution
    
    def print_results(self, metrics: Dict[str, Any]):
        """Pretty print evaluation results"""
        print("\n" + "="*70)
        print("📊 ACCURACY EVALUATION RESULTS")
        print("="*70)
        
        print(f"\n✓ Total Questions Evaluated: {metrics['total_questions']}")
        
        print(f"\n{'─'*70}")
        print("ACCURACY METRICS")
        print(f"{'─'*70}")
        print(f"Top-1 Accuracy: {metrics['top_1_accuracy']:>6.2f}%  ({metrics['top_1_correct']}/{metrics['total_questions']}) ⭐")
        print(f"Top-3 Accuracy: {metrics['top_3_accuracy']:>6.2f}%  ({metrics['top_3_correct']}/{metrics['total_questions']}) ⭐")
        print(f"Top-5 Accuracy: {metrics['top_5_accuracy']:>6.2f}%  ({metrics['top_5_correct']}/{metrics['total_questions']})")
        
        print(f"\n{'─'*70}")
        print("RANK DISTRIBUTION (Where answers were found)")
        print(f"{'─'*70}")
        dist = metrics['rank_distribution']
        for rank in [1, 2, 3, 4, 5]:
            print(f"Rank {rank}: {dist[rank]:>3} questions")
        print(f"Not Found: {dist['not_found']:>3} questions")
        
        print("\n" + "="*70)
        
        # Evaluation
        if metrics['top_1_accuracy'] >= 80:
            print("✅ EXCELLENT: Top-1 accuracy ≥ 80%")
        elif metrics['top_1_accuracy'] >= 60:
            print("✓ GOOD: Top-1 accuracy ≥ 60%")
        else:
            print("⚠ NEEDS IMPROVEMENT: Top-1 accuracy < 60%")
        
        if metrics['top_3_accuracy'] >= 90:
            print("✅ EXCELLENT: Top-3 accuracy ≥ 90%")
        elif metrics['top_3_accuracy'] >= 75:
            print("✓ GOOD: Top-3 accuracy ≥ 75%")
        else:
            print("⚠ NEEDS IMPROVEMENT: Top-3 accuracy < 75%")
        
        print("="*70)
    
    def save_results(self, filename: str = 'accuracy_results.json'):
        """Save evaluation results"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.calculate_metrics(),
            'detailed_results': self.evaluation_results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")


# ==================== GROUND TRUTH TEST CASES ====================
# These are carefully crafted with expected answers from your PDFs

# ==================== GROUND TRUTH TEST CASES (Max Estates 128) ====================
ACCURACY_TEST_CASES = [
    # ============ METADATA & GENERAL INFO ============
    {
        'question': 'What about rail connectivity with daily trains?',
        'expected_answer': ' Shatabdi Express, Jan Shatabdi Express and Nanda Devi Express',
        'category': 'metadata'
    },
    {
        'question': 'How many total residences are there in the 222 Rajpur community?',
        'expected_answer': '22 bespoke residences',
        'category': 'metadata'
    },
    {
        'question': 'What is the RERA Registration Number for 222 Rajpur?',
        'expected_answer': 'UKREP08170000001',
        'category': 'metadata'
    },
    {
        'question': 'How many Townhouse units are available?',
        'expected_answer': '14 Units',
        'category': 'metadata'
    },
    {
        'question': 'Which forest does the project abut?',
        'expected_answer': 'Malsi Reserve Forest',
        'category': 'metadata'
    },

    # ============ NUMERICAL / DIMENSIONS ============
    {
        'question': 'What is the distance to Max Super Speciality Hospital?',
        'expected_answer': '3.2 km',
        'category': 'numerical'
    },
    {
        'question': 'How far is the Jolly Grant Airport from the project?',
        'expected_answer': '36.9 km (approx 75 minutes)',
        'category': 'numerical'
    },
    {
        'question': 'What is the Built-up Area of a Townhouse unit?',
        'expected_answer': '3990 sq.ft.',
        'category': 'numerical'
    },
    {
        'question': 'What is the Carpet Area of a Courtyard Villa?',
        'expected_answer': '3276 sq.ft.',
        'category': 'numerical'
    },
    {
        'question': 'What is the plot size range for Forest Villas?',
        'expected_answer': '435-499 sq.m.',
        'category': 'numerical'
    },
    {
        'question': 'What is the ceiling height provided in the residences?',
        'expected_answer': '10.5 ft',
        'category': 'numerical'
    },
    {
        'question': 'How many car parking spaces are provided per residence?',
        'expected_answer': '2 car parking spaces',
        'category': 'numerical'
    },
    {
        'question': 'What is the booking amount percentage in the Down Payment Plan?',
        'expected_answer': '10% of Basic Sales Price',
        'category': 'numerical'
    },

    # ============ CONTENT / SPECIFICATIONS ============
    {
        'question': 'What brand of sanitary fixtures is provided in the bathrooms?',
        'expected_answer': 'Kohler or equivalent',
        'category': 'content'
    },
    {
        'question': 'What type of windows are installed in the residences?',
        'expected_answer': 'Insulated premium windows with double glazed units',
        'category': 'content'
    },
    {
        'question': 'What flooring is used in the bedrooms?',
        'expected_answer': 'Perlato marble flooring',
        'category': 'content'
    },
    {
        'question': 'What brand of microwave and oven is provided in the kitchen?',
        'expected_answer': 'Siemens or equivalent',
        'category': 'content'
    },
    {
        'question': 'What security systems are available in the community?',
        'expected_answer': '24x7 monitoring, CCTV cameras, and RFID-based car recognition',
        'category': 'content'
    }
]


# ==================== RUN EVALUATION ====================
if __name__ == "__main__":
    evaluator = AccuracyEvaluator()
    
    print("\n🎯 Starting Accuracy Evaluation...")
    print("This will test if the system retrieves CORRECT information\n")
    
    # Run evaluation
    metrics = evaluator.run_evaluation(ACCURACY_TEST_CASES)
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    evaluator.save_results('baseline_accuracy.json')
    
    # Also print category-wise breakdown
    print("\n" + "="*70)
    print("📋 CATEGORY-WISE BREAKDOWN")
    print("="*70)
    
    categories = {}
    for result in evaluator.evaluation_results:
        # Find category from test case
        test_case = next((tc for tc in ACCURACY_TEST_CASES if tc['question'] == result['question']), None)
        if test_case:
            cat = test_case.get('category', 'unknown')
            if cat not in categories:
                categories[cat] = {'total': 0, 'top_1': 0, 'top_3': 0}
            
            categories[cat]['total'] += 1
            if result.get('top_1'):
                categories[cat]['top_1'] += 1
            if result.get('top_3'):
                categories[cat]['top_3'] += 1
    
    for category, stats in categories.items():
        total = stats['total']
        top_1_acc = round(stats['top_1'] / total * 100, 1) if total > 0 else 0
        top_3_acc = round(stats['top_3'] / total * 100, 1) if total > 0 else 0
        
        print(f"\n{category.upper()}:")
        print(f"  Top-1: {top_1_acc}% ({stats['top_1']}/{total})")
        print(f"  Top-3: {top_3_acc}% ({stats['top_3']}/{total})")
    
    print("="*70)