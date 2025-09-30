"""Data models for portfolio scanner"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

@dataclass
class RepositoryData:
    name: str
    full_name: str
    description: str
    url: str
    stars: int
    forks: int
    language: Optional[str]
    topics: List[str]
    created_at: str
    updated_at: str
    frameworks: Dict[str, List]
    code_analysis: Dict[str, Any]
    ai_ml_relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PortfolioData:
    def __init__(self):
        self.total_repositories: int = 0
        self.generation_date: str = ""
        self.repositories: List[RepositoryData] = []
        self.insights: Dict[str, Any] = {}
        self.categories: Dict[str, List[Dict]] = {}
        self.skills: Dict[str, Any] = {}
        self.highlights: List[Dict[str, Any]] = []
        self.expertise_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_repositories': self.total_repositories,
            'generation_date': self.generation_date,
            'repositories': [r.to_dict() for r in self.repositories],
            'insights': self.insights,
            'categories': self.categories,
            'skills': self.skills,
            'highlights': self.highlights,
            'expertise_metrics': self.expertise_metrics
        }
