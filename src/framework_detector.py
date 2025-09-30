"""Framework detection from repository contents"""
from typing import Dict, List, Any

class FrameworkDetector:
    def __init__(self):
        self.frameworks = {
            'ml_frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'keras'],
            'llm_frameworks': ['transformers', 'openai', 'langchain', 'anthropic'],
            'data_frameworks': ['pandas', 'numpy', 'dask']
        }
    
    async def detect_frameworks(self, repo_data: Dict) -> Dict[str, List[str]]:
        detected = {k: [] for k in self.frameworks.keys()}
        
        description = (repo_data.get('description') or '').lower()
        topics = [t.lower() for t in repo_data.get('topics', [])]
        text = description + ' ' + ' '.join(topics)
        
        for category, frameworks in self.frameworks.items():
            for framework in frameworks:
                if framework in text:
                    detected[category].append(framework)
        
        return detected
