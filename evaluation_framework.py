# ==================== evaluation_framework.py ====================
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import ndcg_score
from collections import defaultdict
import time
import re 
class RAGEvaluator:
    """Comprehensive RAG System Evaluator"""
    
    def __init__(self, document_system, ground_truth_file: str):
        self.system = document_system
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
            self.questions = data['questions']
        
        # Results storage
        self.results = []
        self.metrics = {}
        
    def evaluate_all(self, k_values=[1, 3, 5, 10]):
        """Run complete evaluation"""
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE RAG EVALUATION")
        print("="*70)
        
        print(f"\n📊 Evaluating {len(self.questions)} questions...")
        
        for i, q in enumerate(self.questions, 1):
            print(f"\n[{i}/{len(self.questions)}] Testing: {q['question'][:60]}...")
            result = self._evaluate_single_question(q, k_values)
            self.results.append(result)
            time.sleep(16)
        # Calculate all metrics
        self._calculate_metrics(k_values)
        
        # Print summary
        self._print_summary()
        
        # Save detailed results
        self._save_results()
        
        return self.metrics
    
    def _evaluate_single_question(self, q: Dict, k_values: List[int]) -> Dict:
        """Evaluate single question"""
        question = q['question']
        expected_answer = q.get('answer', '')
        required_entities = q.get('required_entities', [])
        section = q.get('section', 'Unknown')
        
        # Time the query
        start_time = time.time()
        
        # Get search results
        max_k = max(k_values)
        search_results = self.system.hybrid_search(question, k=max_k)
        
        search_latency = (time.time() - start_time) * 1000
        
        # Get LLM answer
        llm_start = time.time()
        llm_result = self.system.generate_answer_with_llm(
            question, 
            search_results, 
            max_chunks=5,
            stream=False
        )
        llm_latency = (time.time() - llm_start) * 1000
        llm_answer = llm_result.get('answer')

        
        total_latency = (time.time() - start_time) * 1000
        if llm_answer is None:
            llm_answer = ""
        
        # Find rank of correct answer
        correct_rank = self._find_correct_rank(search_results, q)
        
        # Calculate metrics for this question
        result = {
            'question_id': q.get('id', ''),
            'section': section,
            'question': question,
            'expected_answer': expected_answer,
            'llm_answer': llm_answer,
            'correct_rank': correct_rank,
            'search_latency_ms': search_latency,
            'llm_latency_ms': llm_latency,
            'total_latency_ms': total_latency,
            'retrieved_chunks': len(search_results),
        }
        
        # Recall@K for each k
        for k in k_values:
            result[f'recall_at_{k}'] = 1.0 if correct_rank > 0 and correct_rank <= k else 0.0
        
        # Top-1 and Top-3 accuracy
        result['top_1_correct'] = 1.0 if correct_rank == 1 else 0.0
        result['top_3_correct'] = 1.0 if correct_rank > 0 and correct_rank <= 3 else 0.0
        
        # Reciprocal rank
        result['reciprocal_rank'] = 1.0 / correct_rank if correct_rank > 0 else 0.0
        
        # Entity coverage
        result['entity_coverage'] = self._calculate_entity_coverage(
            llm_answer, required_entities
        )
        
        # Hallucination check (for non-negative queries)
        if section != 'G':  # G is negative queries
            result['is_hallucination'] = self._check_hallucination(
                llm_answer, search_results[:5], expected_answer
            )
        else:
            # For negative queries, check if system correctly said "not found"
            result['correctly_rejected'] = self._is_correctly_rejected(llm_answer)
        
        return result
    
    def _find_correct_rank(self, search_results: List, question_data: Dict) -> int:
        required_entities = question_data.get('required_entities', [])
        
        if not required_entities:
            return 1
            
        for rank, (doc, score) in enumerate(search_results, 1):
            chunk_text = doc.page_content.lower()
            chunk_clean = re.sub(r'[^a-z0-9\s]', '', chunk_text)
            
            for entity in required_entities:
                entity_clean = re.sub(r'[^a-z0-9\s]', '', entity.lower())
                if entity_clean and entity_clean in chunk_clean:
                    return rank
                    
        return 0
    
    def _calculate_entity_coverage(self, answer: str, required_entities: List[str]) -> float:
        """Calculate what % of required entities are in answer"""
        if not required_entities:
            return 1.0
        
        answer_lower = answer.lower()
        found_count = sum(1 for entity in required_entities if entity.lower() in answer_lower)
        
        return found_count / len(required_entities)
    
    def _check_hallucination(self, answer: str, top_chunks: List, expected: str) -> bool:
        if "not found" in answer.lower() or "no information" in answer.lower():
            return False
        
        stop_words = {'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'what', 
                      'which', 'this', 'that', 'these', 'those', 'then', 'so', 'than', 
                      'such', 'both', 'for', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'to', 'of', 'in', 
                      'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through', 
                      'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down', 'out', 
                      'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                      'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                      'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                      'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}
        
        answer_lower = answer.lower()
        answer_clean = re.sub(r'[^a-z0-9\s]', '', answer_lower)
        answer_words = set(answer_clean.split()) - stop_words
        
        if not answer_words:
            return False
            
        for doc, _ in top_chunks:
            chunk_clean = re.sub(r'[^a-z0-9\s]', '', doc.page_content.lower())
            chunk_words = set(chunk_clean.split())
            overlap = len(answer_words & chunk_words)
            
            if overlap > len(answer_words) * 0.3:
                return False
        
        return True
    
    def _is_correctly_rejected(self, answer: str) -> bool:
        """Check if system correctly said 'not found' for negative query"""
        if not answer:
            return False
        rejection_phrases = [
            "not found",
            "no information",
            "cannot find",
            "not available",
            "not mentioned",
            "does not contain"
        ]
        
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in rejection_phrases)
    
    def _calculate_metrics(self, k_values: List[int]):
        """Calculate aggregate metrics"""
        total = len(self.results)
        
        # Recall@K
        for k in k_values:
            recall = np.mean([r[f'recall_at_{k}'] for r in self.results])
            self.metrics[f'recall_at_{k}'] = recall
        
        # Top-1 and Top-3 Accuracy
        self.metrics['top_1_accuracy'] = np.mean([r['top_1_correct'] for r in self.results])
        self.metrics['top_3_accuracy'] = np.mean([r['top_3_correct'] for r in self.results])
        
        # MRR (Mean Reciprocal Rank)
        self.metrics['mrr'] = np.mean([r['reciprocal_rank'] for r in self.results])
        
        # nDCG@10
        self.metrics['ndcg_at_10'] = self._calculate_ndcg(10)
        
        # Entity Coverage
        entity_scores = [r['entity_coverage'] for r in self.results if 'entity_coverage' in r]
        self.metrics['avg_entity_coverage'] = np.mean(entity_scores) if entity_scores else 0.0
        
        # Hallucination Rate
        hallucinations = [r for r in self.results if r.get('is_hallucination', False)]
        self.metrics['hallucination_rate'] = len(hallucinations) / total
        
        # False Positive Rate (Section G - negative queries)
        section_g = [r for r in self.results if r['section'] == 'G']
        if section_g:
            correctly_rejected = sum(r.get('correctly_rejected', 0) for r in section_g)
            self.metrics['false_positive_rate'] = 1.0 - (correctly_rejected / len(section_g))
        else:
            self.metrics['false_positive_rate'] = 0.0
        
        # Latency metrics
        latencies = [r['total_latency_ms'] for r in self.results]
        self.metrics['avg_latency_ms'] = np.mean(latencies)
        self.metrics['p95_latency_ms'] = np.percentile(latencies, 95)
        self.metrics['p99_latency_ms'] = np.percentile(latencies, 99)
        
        # Latency breakdown
        search_latencies = [r['search_latency_ms'] for r in self.results]
        llm_latencies = [r['llm_latency_ms'] for r in self.results]
        
        self.metrics['avg_search_latency_ms'] = np.mean(search_latencies)
        self.metrics['avg_llm_latency_ms'] = np.mean(llm_latencies)
        
        # Paraphrase robustness (Section F)
        self.metrics['paraphrase_robustness'] = self._calculate_paraphrase_robustness()
    
    def _calculate_ndcg(self, k: int) -> float:
        """Calculate nDCG@k"""
        ndcg_scores = []
        
        for r in self.results:
            rank = r['correct_rank']
            
            if rank == 0:
                # Not found = 0 score
                ndcg_scores.append(0.0)
            else:
                # Create relevance vector
                relevance = [0] * k
                if rank <= k:
                    relevance[rank - 1] = 1  # Correct answer is relevant
                
                # Ideal relevance (best case)
                ideal_relevance = [1] + [0] * (k - 1)
                
                # Calculate DCG
                try:
                    score = ndcg_score([ideal_relevance], [relevance])
                    ndcg_scores.append(score)
                except:
                    ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores)
    
    def _calculate_paraphrase_robustness(self) -> float:
        """Calculate consistency across paraphrases (Section F)"""
        section_f = [r for r in self.results if r['section'] == 'F']
        
        if not section_f:
            return 1.0
        
        # Group by question variant
        # Assuming question IDs like F1a, F1b, F1c are paraphrases
        groups = defaultdict(list)
        for r in section_f:
            q_id = r['question_id']
            base_id = q_id[:-1] if q_id[-1].isalpha() else q_id
            groups[base_id].append(r)
        
        # Calculate consistency within each group
        consistencies = []
        for group_results in groups.values():
            if len(group_results) < 2:
                continue
            
            # Check if all variants found answer at same rank
            ranks = [r['correct_rank'] for r in group_results]
            # Consistency = 1 if all same, 0 if all different
            unique_ranks = len(set(ranks))
            consistency = 1.0 if unique_ranks == 1 else (1.0 / unique_ranks)
            consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0
    
    def _print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("📊 EVALUATION SUMMARY")
        print("="*70)
        
        print("\n🎯 ACCURACY METRICS:")
        print(f"  Recall@1:  {self.metrics['recall_at_1']:.2%} (Target: ≥75%)")
        print(f"  Recall@3:  {self.metrics['recall_at_3']:.2%} (Target: ≥90%)")
        print(f"  Recall@5:  {self.metrics['recall_at_5']:.2%}")
        print(f"  Recall@10: {self.metrics['recall_at_10']:.2%}")
        
        print(f"\n  Top-1 Accuracy: {self.metrics['top_1_accuracy']:.2%}")
        print(f"  Top-3 Accuracy: {self.metrics['top_3_accuracy']:.2%}")
        
        print(f"\n  MRR:        {self.metrics['mrr']:.4f}")
        print(f"  nDCG@10:    {self.metrics['ndcg_at_10']:.4f}")
        
        print(f"\n📝 QUALITY METRICS:")
        print(f"  Entity Coverage:      {self.metrics['avg_entity_coverage']:.2%}")
        print(f"  Paraphrase Robustness: {self.metrics['paraphrase_robustness']:.2%}")
        print(f"  Hallucination Rate:    {self.metrics['hallucination_rate']:.2%} (Target: <5%)")
        print(f"  False Positive Rate:   {self.metrics['false_positive_rate']:.2%} (Target: 0%)")
        
        print(f"\n⚡ LATENCY METRICS:")
        print(f"  Average:    {self.metrics['avg_latency_ms']:.0f}ms")
        print(f"  P95:        {self.metrics['p95_latency_ms']:.0f}ms")
        print(f"  P99:        {self.metrics['p99_latency_ms']:.0f}ms")
        
        print(f"\n  Search:     {self.metrics['avg_search_latency_ms']:.0f}ms")
        print(f"  LLM:        {self.metrics['avg_llm_latency_ms']:.0f}ms")
        
        print("\n" + "="*70)
    
    def _save_results(self):
        """Save detailed results to file"""
        output = {
            'metrics': self.metrics,
            'detailed_results': self.results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n💾 Detailed results saved to: evaluation_results.json")



if __name__ == "__main__":
    from document_system import HybridDocumentSystem
    
    print("\n🚀 Initializing Document System...")
    system = HybridDocumentSystem()
    
    print("\n📈 Loading Evaluator...")
    # Make sure 'ground_truth.json' is in the same folder
    evaluator = RAGEvaluator(system, 'ground_truth.json')
    
    print("\n🔥 Starting Full Evaluation...")
    # This is the line that actually triggers the whole process!
    evaluator.evaluate_all()