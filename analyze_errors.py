import json
from collections import defaultdict

def analyze_evaluation(json_file='evaluation_results.json'):
    print(f"🔍 Loading results from {json_file}...\n")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Could not find {json_file}. Make sure the file name is correct.")
        return

    results = data.get('detailed_results', [])
    if not results:
        print("❌ No detailed results found in the JSON.")
        return

    # Group by section
    sections = defaultdict(list)
    for res in results:
        section = res.get('section', 'Unknown')
        sections[section].append(res)

    print("="*80)
    print("📊 SECTION-BY-SECTION FAILURE ANALYSIS")
    print("="*80)

    for section, reqs in sorted(sections.items()):
        print(f"\n" + "-"*80)
        print(f"📑 SECTION {section} ({len(reqs)} questions)")
        print("-"*80)
        
        # Calculate section metrics
        avg_recall_1 = sum(r.get('recall_at_1', 0) for r in reqs) / len(reqs)
        avg_entity = sum(r.get('entity_coverage', 0) for r in reqs) / len(reqs)
        hallucinations = sum(1 for r in reqs if r.get('is_hallucination', False))
        
        print(f"Overall Health: Recall@1: {avg_recall_1:.0%} | Entity Coverage: {avg_entity:.0%} | Hallucinations: {hallucinations}/{len(reqs)}")
        print("\n⚠️  FAILED QUESTIONS IN THIS SECTION:")
        
        failure_found = False
        
        for r in reqs:
            q_id = r.get('question_id', 'Unknown')
            question = r.get('question', 'Unknown')
            rank = r.get('correct_rank', 0)
            is_hallucination = r.get('is_hallucination', False)
            entity_cov = r.get('entity_coverage', 1.0)
            section_type = r.get('section', '')
            
            issues = []
            
            # Check for ranking failures
            if rank == 0:
                issues.append("Rank 0 (Not found in top 10 chunks)")
            elif rank > 3:
                issues.append(f"Poor Rank (Found at rank {rank})")
                
            # Check for hallucination (skip for Negative queries section G)
            if section_type != 'G' and is_hallucination:
                issues.append("Hallucination Detected")
                
            # Check for negative query failures (Section G)
            if section_type == 'G' and not r.get('correctly_rejected', True):
                issues.append("Failed to Reject (False Positive)")
                
            # Check for entity misses
            if entity_cov < 1.0:
                issues.append(f"Missing Entities (Coverage: {entity_cov:.0%})")
                
            # If any issues found, print them
            if issues:
                failure_found = True
                print(f"  [{q_id}] {question[:60]}...")
                print(f"      ↳ Issues: {', '.join(issues)}")
                
        if not failure_found:
            print("  ✅ All questions in this section passed perfectly!")

if __name__ == "__main__":
    analyze_evaluation('evaluation_results.json')