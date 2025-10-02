"""Data models for portfolio generation"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    code_analysis: Dict
    ai_ml_relevance_score: float
    project_status: str = 'past'  # 'current', 'recent', or 'past'
    
    def to_dict(self):
        return {
            'name': self.name,
            'full_name': self.full_name,
            'description': self.description,
            'url': self.url,
            'stars': self.stars,
            'forks': self.forks,
            'language': self.language,
            'topics': self.topics,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'frameworks': self.frameworks,
            'code_analysis': self.code_analysis,
            'ai_ml_relevance_score': self.ai_ml_relevance_score,
            'project_status': self.project_status
        }


class PortfolioData:
    def __init__(self):
        self.total_repositories: int = 0
        self.generation_date: str = ""
        self.repositories: List[RepositoryData] = []
        self.insights: Dict = {}
        self.categories: Dict[str, List[Dict]] = {}
        self.skills: Dict = {}
        self.highlights: List[Dict] = []
        self.expertise_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict:
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
