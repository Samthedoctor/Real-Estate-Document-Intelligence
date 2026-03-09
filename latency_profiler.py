# ==================== latency_profiler.py ====================
import time
from typing import Dict, Any
from contextlib import contextmanager

class LatencyProfiler:
    """Profile stage-by-stage latency"""
    
    def __init__(self):
        self.stages = {}
        self.current_stage = None
        self.start_time = None
    
    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager to profile a stage"""
        start = time.time()
        try:
            yield
        finally:
            elapsed = (time.time() - start) * 1000
            if stage_name not in self.stages:
                self.stages[stage_name] = []
            self.stages[stage_name].append(elapsed)
    
    def get_breakdown(self) -> Dict[str, float]:
        """Get average breakdown"""
        breakdown = {}
        for stage, times in self.stages.items():
            breakdown[f"{stage}_ms"] = sum(times) / len(times) if times else 0
        return breakdown
    
    def reset(self):
        """Reset profiler"""
        self.stages = {}
    
    def print_breakdown(self):
        """Print latency breakdown"""
        breakdown = self.get_breakdown()
        
        print("\n⚡ LATENCY BREAKDOWN:")
        print("="*50)
        
        total = sum(breakdown.values())
        
        for stage, latency in sorted(breakdown.items()):
            pct = (latency / total * 100) if total > 0 else 0
            print(f"  {stage:.<40} {latency:>6.1f}ms ({pct:>5.1f}%)")
        
        print("="*50)
        print(f"  {'TOTAL':.<40} {total:>6.1f}ms (100.0%)")
        print()