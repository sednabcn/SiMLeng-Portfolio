"""
Data models for AI Portfolio Scanner and portfolio generation

Combines:
- Original RepositoryData and PortfolioData models
- ScanConfig and supporting models for main.py integration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime


# ============================================================================
# ORIGINAL MODELS (Preserved from existing data_models.py)
# ============================================================================

@dataclass
class RepositoryData:
    """Original repository data model for portfolio generation"""
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
    pushed_at: str = ''  # Last push date - most accurate for activity tracking
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
            'pushed_at': self.pushed_at,
            'frameworks': self.frameworks,
            'code_analysis': self.code_analysis,
            'ai_ml_relevance_score': self.ai_ml_relevance_score,
            'project_status': self.project_status
        }


class PortfolioData:
    """Original portfolio data model"""
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


# ============================================================================
# NEW MODELS (Required for main.py and workflow integration)
# ============================================================================

@dataclass
class ScanConfig:
    """
    Configuration for portfolio scanning (required by main.py)
    
    This class bridges the gap between CLI arguments and the existing
    portfolio generation system.
    """
    github_token: str
    target_user: Optional[str] = None
    target_repos: Optional[List[str]] = None
    frameworks: Dict[str, List[str]] = field(default_factory=dict)
    analysis_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path("portfolio-output"))
    
    def __post_init__(self):
        """Validate and process configuration"""
        if not self.target_user and not self.target_repos:
            raise ValueError("Either target_user or target_repos must be specified")
        
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Set default analysis config if not provided
        if not self.analysis_config:
            self.analysis_config = {
                'max_repos': 50,
                'max_file_size': 1048576,
                'include_forks': False,
                'min_stars': 0,
                'exclude_archived': True
            }
        
        # Set default output config if not provided
        if not self.output_config:
            self.output_config = {
                'format': 'json',
                'generate_html': True,
                'generate_markdown': True,
                'include_code_samples': True,
                'max_code_sample_lines': 50
            }
    
    def get_max_repos(self) -> int:
        """Helper to get max repos setting"""
        return self.analysis_config.get('max_repos', 50)
    
    def should_include_forks(self) -> bool:
        """Helper to check if forks should be included"""
        return self.analysis_config.get('include_forks', False)
    
    def get_min_stars(self) -> int:
        """Helper to get minimum stars threshold"""
        return self.analysis_config.get('min_stars', 0)


@dataclass
class RepositoryInfo:
    """
    Extended repository information for analysis
    (Complements RepositoryData for more detailed scanning)
    """
    name: str
    full_name: str
    description: Optional[str]
    url: str
    stars: int
    forks: int
    language: Optional[str]
    created_at: datetime
    updated_at: datetime
    size: int
    topics: List[str] = field(default_factory=list)
    is_fork: bool = False
    is_archived: bool = False
    default_branch: str = "main"
    
    def to_repository_data(self, frameworks: Dict, code_analysis: Dict, 
                          relevance_score: float, pushed_at: str = '',
                          status: str = 'past') -> RepositoryData:
        """Convert to original RepositoryData format"""
        return RepositoryData(
            name=self.name,
            full_name=self.full_name,
            description=self.description or "",
            url=self.url,
            stars=self.stars,
            forks=self.forks,
            language=self.language,
            topics=self.topics,
            created_at=self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            updated_at=self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            frameworks=frameworks,
            code_analysis=code_analysis,
            ai_ml_relevance_score=relevance_score,
            pushed_at=pushed_at,
            project_status=status
        )


@dataclass
class FrameworkDetection:
    """Framework detection result"""
    framework_name: str
    category: str
    file_path: str
    line_number: Optional[int] = None
    context: Optional[str] = None
    confidence: float = 1.0


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    file_path: str
    file_type: str
    size: int
    language: Optional[str]
    frameworks_detected: List[FrameworkDetection] = field(default_factory=list)
    complexity_score: Optional[float] = None
    quality_score: Optional[float] = None


@dataclass
class RepositoryAnalysis:
    """
    Complete analysis results for a repository
    (Bridge between scanning and portfolio generation)
    """
    repository: RepositoryInfo
    files_analyzed: List[FileAnalysis] = field(default_factory=list)
    frameworks_found: Dict[str, List[str]] = field(default_factory=dict)
    total_files: int = 0
    total_code_lines: int = 0
    is_ai_ml_relevant: bool = False
    relevance_score: float = 0.0
    expertise_score: float = 0.0
    primary_language: Optional[str] = None
    
    def add_file_analysis(self, file_analysis: FileAnalysis):
        """Add file analysis to repository analysis"""
        self.files_analyzed.append(file_analysis)
        self.total_files += 1
        
        # Aggregate frameworks
        for detection in file_analysis.frameworks_detected:
            category = detection.category
            framework = detection.framework_name
            
            if category not in self.frameworks_found:
                self.frameworks_found[category] = []
            
            if framework not in self.frameworks_found[category]:
                self.frameworks_found[category].append(framework)
        
        # Update relevance
        if file_analysis.frameworks_detected:
            self.is_ai_ml_relevant = True
    
    def to_repository_data(self) -> RepositoryData:
        """Convert to original RepositoryData format"""
        # Prepare code analysis summary
        code_analysis = {
            'total_files': self.total_files,
            'total_lines': self.total_code_lines,
            'primary_language': self.primary_language,
            'complexity_score': self.expertise_score,
            'quality_score': self.expertise_score
        }
        
        # Determine project status based on last update
        pushed_at = self.repository.updated_at.isoformat() if isinstance(self.repository.updated_at, datetime) else self.repository.updated_at
        
        return self.repository.to_repository_data(
            frameworks=self.frameworks_found,
            code_analysis=code_analysis,
            relevance_score=self.relevance_score,
            pushed_at=pushed_at,
            status='current'  # Could be enhanced based on activity
        )


@dataclass
class PortfolioStatistics:
    """Overall portfolio statistics"""
    repositories_scanned: int = 0
    repositories_analyzed: int = 0
    relevant_repositories: int = 0
    total_files_analyzed: int = 0
    total_code_lines: int = 0
    frameworks_detected: Dict[str, int] = field(default_factory=dict)
    languages_used: Dict[str, int] = field(default_factory=dict)
    overall_expertise_score: float = 0.0
    total_stars: int = 0
    total_forks: int = 0
    
    def update_from_analysis(self, analysis: RepositoryAnalysis):
        """Update statistics from repository analysis"""
        self.repositories_analyzed += 1
        
        if analysis.is_ai_ml_relevant:
            self.relevant_repositories += 1
        
        self.total_files_analyzed += analysis.total_files
        self.total_code_lines += analysis.total_code_lines
        self.total_stars += analysis.repository.stars
        self.total_forks += analysis.repository.forks
        
        # Aggregate frameworks
        for category, frameworks in analysis.frameworks_found.items():
            for framework in frameworks:
                self.frameworks_detected[framework] = \
                    self.frameworks_detected.get(framework, 0) + 1
        
        # Aggregate languages
        if analysis.primary_language:
            self.languages_used[analysis.primary_language] = \
                self.languages_used.get(analysis.primary_language, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'repositories_scanned': self.repositories_scanned,
            'repositories_analyzed': self.repositories_analyzed,
            'relevant_repositories': self.relevant_repositories,
            'total_files_analyzed': self.total_files_analyzed,
            'total_code_lines': self.total_code_lines,
            'frameworks_detected': self.frameworks_detected,
            'languages_used': self.languages_used,
            'overall_expertise_score': self.overall_expertise_score,
            'total_stars': self.total_stars,
            'total_forks': self.total_forks
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_analysis_to_portfolio_data(
    analyses: List[RepositoryAnalysis],
    statistics: PortfolioStatistics,
    generation_date: str
) -> PortfolioData:
    """
    Convert new-style RepositoryAnalysis to original PortfolioData format
    
    This function bridges the gap between the scanning system and the
    existing portfolio generation system.
    """
    portfolio = PortfolioData()
    portfolio.total_repositories = len(analyses)
    portfolio.generation_date = generation_date
    
    # Convert each analysis to RepositoryData
    for analysis in analyses:
        repo_data = analysis.to_repository_data()
        portfolio.repositories.append(repo_data)
    
    # Populate expertise metrics from statistics
    portfolio.expertise_metrics = {
        'overall_score': statistics.overall_expertise_score,
        'total_stars': statistics.total_stars,
        'total_forks': statistics.total_forks,
        'ai_ml_relevant': statistics.relevant_repositories,
        'total_repositories': statistics.repositories_analyzed
    }
    
    # Populate skills from frameworks detected
    portfolio.skills = {
        'frameworks': statistics.frameworks_detected,
        'languages': statistics.languages_used
    }
    
    # Generate insights
    portfolio.insights = {
        'frameworks_count': len(statistics.frameworks_detected),
        'languages_count': len(statistics.languages_used),
        'avg_expertise_score': statistics.overall_expertise_score,
        'ai_ml_percentage': (
            (statistics.relevant_repositories / statistics.repositories_analyzed * 100)
            if statistics.repositories_analyzed > 0 else 0
        )
    }
    
    # Categorize repositories
    portfolio.categories = categorize_repositories(portfolio.repositories)
    
    # Generate highlights
    portfolio.highlights = generate_highlights(portfolio.repositories, statistics)
    
    return portfolio


def categorize_repositories(repositories: List[RepositoryData]) -> Dict[str, List[Dict]]:
    """Categorize repositories by frameworks and topics"""
    categories = {
        'ml_frameworks': [],
        'llm_frameworks': [],
        'data_science': [],
        'deep_learning': [],
        'nlp': [],
        'computer_vision': [],
        'reinforcement_learning': []
    }
    
    for repo in repositories:
        repo_dict = repo.to_dict()
        
        # Categorize based on frameworks
        frameworks = repo.frameworks
        
        if any(fw in frameworks.get('llm_frameworks', []) for fw in ['transformers', 'langchain', 'openai']):
            categories['llm_frameworks'].append(repo_dict)
        
        if any(fw in frameworks.get('ml_frameworks', []) for fw in ['scikit-learn', 'xgboost']):
            categories['ml_frameworks'].append(repo_dict)
        
        if any(fw in frameworks.get('dl_frameworks', []) for fw in ['tensorflow', 'pytorch']):
            categories['deep_learning'].append(repo_dict)
        
        if any(fw in frameworks.get('data_frameworks', []) for fw in ['pandas', 'numpy']):
            categories['data_science'].append(repo_dict)
    
    return categories


def generate_highlights(repositories: List[RepositoryData], 
                       statistics: PortfolioStatistics) -> List[Dict]:
    """Generate portfolio highlights"""
    highlights = []
    
    # Top starred repositories
    top_repos = sorted(repositories, key=lambda r: r.stars, reverse=True)[:5]
    if top_repos:
        highlights.append({
            'type': 'top_starred',
            'title': 'Most Popular Projects',
            'repositories': [r.to_dict() for r in top_repos]
        })
    
    # Most recent projects
    recent_repos = sorted(repositories, key=lambda r: r.pushed_at or r.updated_at, reverse=True)[:5]
    if recent_repos:
        highlights.append({
            'type': 'recent',
            'title': 'Recent Work',
            'repositories': [r.to_dict() for r in recent_repos]
        })
    
    # AI/ML focused projects
    ai_ml_repos = [r for r in repositories if r.ai_ml_relevance_score > 0.5]
    if ai_ml_repos:
        highlights.append({
            'type': 'ai_ml',
            'title': 'AI/ML Projects',
            'repositories': [r.to_dict() for r in ai_ml_repos[:5]]
        })
    
    return highlights
