#!/usr/bin/env python3
"""
Portfolio Builder Module
Builds comprehensive AI/ML portfolio from analyzed repositories

This module provides functionality to:
- Analyze and categorize AI/ML repositories
- Generate portfolio statistics and insights
- Export portfolio data in multiple formats (HTML, Markdown, JSON)
- Calculate expertise metrics and skill proficiency levels

Example:
    >>> builder = PortfolioBuilder()
    >>> portfolio = builder.build_portfolio(repositories, config)
    >>> outputs = builder.export_portfolio(portfolio, formats=['html', 'markdown', 'json'])
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from jinja2 import Template

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class PortfolioData:
    """
    Data class for portfolio information.
    
    Attributes:
        total_repositories: Total number of repositories analyzed
        generation_date: ISO format timestamp of portfolio generation
        repositories: List of analyzed repository objects
        insights: Dictionary of analytical insights
        categories: Categorized repositories by domain
        skills: Extracted skills and technologies
        highlights: Top portfolio highlights
        expertise_metrics: Calculated expertise scores across domains
    """
    total_repositories: int = 0
    generation_date: str = ""
    repositories: List[Any] = field(default_factory=list)
    insights: Dict[str, Any] = field(default_factory=dict)
    categories: Dict[str, List[Dict]] = field(default_factory=dict)
    skills: Dict[str, Any] = field(default_factory=dict)
    highlights: List[Dict[str, Any]] = field(default_factory=list)
    expertise_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_repositories': self.total_repositories,
            'generation_date': self.generation_date,
            'repositories': [
                repo.to_dict() if hasattr(repo, 'to_dict') else asdict(repo) 
                for repo in self.repositories
            ],
            'insights': self.insights,
            'categories': self.categories,
            'skills': self.skills,
            'highlights': self.highlights,
            'expertise_metrics': self.expertise_metrics
        }


class PortfolioBuilder:
    """
    Builds and formats AI/ML portfolio from repository analysis data.
    
    This class provides comprehensive portfolio building capabilities including
    statistical analysis, categorization, skill extraction, and multi-format export.
    """
    
    # Default configuration values
    DEFAULT_CONFIG = {
        'max_highlights': 5,
        'min_relevance_score': 3.0,
        'max_frameworks_display': 15,
        'top_skills_count': 10,
        'min_stars_for_highlight': 10,
        'min_forks_for_highlight': 5,
        'proficiency_thresholds': {
            'advanced': 0.3,
            'intermediate': 0.15,
            'beginner': 0.0
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PortfolioBuilder.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._validate_config(config or {})
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and merge configuration with defaults.
        
        Args:
            config: User-provided configuration
            
        Returns:
            Merged configuration with defaults
        """
        validated = self.DEFAULT_CONFIG.copy()
        validated.update(config)
        
        # Validate numeric values
        if validated['max_highlights'] < 1:
            self.logger.warning("max_highlights must be >= 1, using default")
            validated['max_highlights'] = self.DEFAULT_CONFIG['max_highlights']
        
        if validated['min_relevance_score'] < 0:
            self.logger.warning("min_relevance_score must be >= 0, using default")
            validated['min_relevance_score'] = self.DEFAULT_CONFIG['min_relevance_score']
        
        return validated
    
    def build_portfolio(self, repositories: List[Any], 
                       config: Optional[Dict[str, Any]] = None) -> PortfolioData:
        """
        Build comprehensive portfolio from analyzed repositories.
        
        Args:
            repositories: List of analyzed repository objects
            config: Optional configuration overrides
            
        Returns:
            PortfolioData object containing complete portfolio analysis
            
        Raises:
            ValueError: If repositories list is invalid
        """
        try:
            if not isinstance(repositories, list):
                raise ValueError("repositories must be a list")
            
            self.logger.info(f"Building portfolio from {len(repositories)} repositories")
            
            if not repositories:
                self.logger.warning("No repositories provided, returning empty portfolio")
                return PortfolioData(generation_date=datetime.now().isoformat())
            
            # Update config if provided
            if config:
                self.config.update(config)
            
            portfolio = PortfolioData()
            
            # Basic metadata
            portfolio.total_repositories = len(repositories)
            portfolio.generation_date = datetime.now().isoformat()
            
            # Filter relevant repositories
            min_score = self.config['min_relevance_score']
            relevant_repos = [
                repo for repo in repositories 
                if hasattr(repo, 'ai_ml_relevance_score') and 
                repo.ai_ml_relevance_score >= min_score
            ]
            
            if not relevant_repos:
                self.logger.warning(
                    f"No repositories meet relevance threshold of {min_score}"
                )
                relevant_repos = repositories  # Use all if none meet threshold
            
            portfolio.repositories = relevant_repos
            
            # Generate all portfolio components
            portfolio.insights = self._generate_insights(relevant_repos)
            portfolio.categories = self._categorize_repositories(relevant_repos)
            portfolio.skills = self._extract_skills(relevant_repos)
            portfolio.highlights = self._select_highlights(
                relevant_repos, 
                max_highlights=self.config['max_highlights']
            )
            portfolio.expertise_metrics = self._calculate_expertise_metrics(relevant_repos)
            
            self.logger.info("Portfolio built successfully")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error building portfolio: {e}", exc_info=True)
            raise
    
    def _generate_insights(self, repositories: List[Any]) -> Dict[str, Any]:
        """
        Generate analytical insights from repository data.
        
        Args:
            repositories: List of repository objects
            
        Returns:
            Dictionary containing various insights and statistics
        """
        insights = {
            'total_stars': 0,
            'total_forks': 0,
            'languages': {},
            'frameworks_summary': {},
            'activity_timeline': [],
            'collaboration_score': 0.0,
            'code_quality_score': 0.0,
            'avg_stars': 0.0,
            'avg_forks': 0.0
        }
        
        if not repositories:
            return insights
        
        # Calculate basic statistics
        insights['total_stars'] = sum(
            getattr(repo, 'stars', 0) for repo in repositories
        )
        insights['total_forks'] = sum(
            getattr(repo, 'forks', 0) for repo in repositories
        )
        insights['avg_stars'] = insights['total_stars'] / len(repositories)
        insights['avg_forks'] = insights['total_forks'] / len(repositories)
        
        # Language distribution
        language_counts = {}
        for repo in repositories:
            lang = getattr(repo, 'language', None)
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        total_repos = len(repositories)
        insights['languages'] = {
            lang: {
                "count": count, 
                "percentage": round((count / total_repos) * 100, 1)
            }
            for lang, count in sorted(
                language_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        }
        
        # Framework summary
        framework_counts = {}
        for repo in repositories:
            frameworks = getattr(repo, 'frameworks', {})
            if not isinstance(frameworks, dict):
                continue
                
            for category, fw_list in frameworks.items():
                if not isinstance(fw_list, list):
                    continue
                    
                for framework in fw_list:
                    name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                    framework_counts[name] = framework_counts.get(name, 0) + 1
        
        insights['frameworks_summary'] = dict(
            sorted(
                framework_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.config['max_frameworks_display']]
        )
        
        # Activity timeline
        activity_by_year = {}
        for repo in repositories:
            created_at = getattr(repo, 'created_at', None)
            if created_at:
                try:
                    year = created_at[:4]  # Extract year from ISO format
                    activity_by_year[year] = activity_by_year.get(year, 0) + 1
                except (IndexError, TypeError):
                    continue
        
        insights['activity_timeline'] = [
            {"year": year, "repositories": count}
            for year, count in sorted(activity_by_year.items())
        ]
        
        # Collaboration score (based on average forks)
        if insights['avg_forks'] > 0:
            # Scale: 0-10, where 10 forks average = score of 10
            insights['collaboration_score'] = min(
                (insights['avg_forks'] / 10.0) * 10, 
                10.0
            )
        
        # Code quality score (based on tests and documentation)
        quality_indicators = 0
        for repo in repositories:
            code_analysis = getattr(repo, 'code_analysis', {})
            if isinstance(code_analysis, dict):
                if code_analysis.get('has_tests'):
                    quality_indicators += 1
                quality_metrics = code_analysis.get('quality_metrics', {})
                if isinstance(quality_metrics, dict):
                    if quality_metrics.get('documentation', 0) > 0:
                        quality_indicators += 0.5
        
        insights['code_quality_score'] = min(
            (quality_indicators / total_repos) * 10, 
            10.0
        )
        
        return insights
    
    def _categorize_repositories(self, repositories: List[Any]) -> Dict[str, List[Dict]]:
        """
        Categorize repositories by AI/ML domain and type.
        
        Args:
            repositories: List of repository objects
            
        Returns:
            Dictionary mapping category names to lists of repository dictionaries
        """
        categories = {
            'machine_learning': [],
            'deep_learning': [],
            'natural_language_processing': [],
            'computer_vision': [],
            'data_science': [],
            'research': [],
            'production': [],
            'educational': []
        }
        
        for repo in repositories:
            try:
                repo_dict = repo.to_dict() if hasattr(repo, 'to_dict') else asdict(repo)
            except Exception as e:
                self.logger.warning(f"Could not convert repo to dict: {e}")
                continue
            
            # Extract repository attributes safely
            frameworks = getattr(repo, 'frameworks', {})
            if not isinstance(frameworks, dict):
                frameworks = {}
            
            description = getattr(repo, 'description', '') or ''
            description = description.lower()
            
            topics = getattr(repo, 'topics', []) or []
            topics = [topic.lower() for topic in topics if isinstance(topic, str)]
            
            code_analysis = getattr(repo, 'code_analysis', {})
            if not isinstance(code_analysis, dict):
                code_analysis = {}
            
            # Categorization logic
            ml_frameworks = frameworks.get('ml_frameworks', [])
            dl_frameworks = frameworks.get('deep_learning', [])
            llm_frameworks = frameworks.get('llm_frameworks', [])
            cv_frameworks = frameworks.get('computer_vision', [])
            data_frameworks = frameworks.get('data_frameworks', [])
            
            # Machine Learning
            ml_indicators = ['scikit-learn', 'xgboost', 'lightgbm', 'catboost']
            if any(fw in str(ml_frameworks) for fw in ml_indicators):
                categories['machine_learning'].append(repo_dict)
            
            # Deep Learning
            dl_indicators = ['tensorflow', 'pytorch', 'keras', 'jax']
            if any(fw in str(ml_frameworks + dl_frameworks) for fw in dl_indicators):
                categories['deep_learning'].append(repo_dict)
            
            # NLP
            nlp_indicators = ['nlp', 'natural language', 'text', 'language model', 
                            'bert', 'gpt', 'transformer']
            if (any(ind in description for ind in nlp_indicators) or 
                any(ind in topics for ind in nlp_indicators) or
                any(fw in str(llm_frameworks) for fw in ['transformers', 'openai', 'langchain'])):
                categories['natural_language_processing'].append(repo_dict)
            
            # Computer Vision
            cv_indicators = ['vision', 'image', 'opencv', 'computer vision', 
                           'cnn', 'detection', 'segmentation', 'yolo']
            if (any(ind in description for ind in cv_indicators) or 
                any(ind in topics for ind in cv_indicators) or
                'opencv' in str(cv_frameworks)):
                categories['computer_vision'].append(repo_dict)
            
            # Data Science
            ds_indicators = ['pandas', 'numpy', 'scipy', 'matplotlib']
            if (any(fw in str(data_frameworks) for fw in ds_indicators) or
                'data science' in description or 'data-science' in topics):
                categories['data_science'].append(repo_dict)
            
            # Research
            research_indicators = ['paper', 'research', 'experiment', 'study', 
                                 'analysis', 'arxiv', 'publication']
            if (any(ind in description for ind in research_indicators) or
                any(ind in topics for ind in research_indicators) or
                code_analysis.get('has_notebooks', False)):
                categories['research'].append(repo_dict)
            
            # Production
            prod_indicators = ['api', 'deploy', 'production', 'docker', 
                             'kubernetes', 'fastapi', 'flask']
            if any(ind in description for ind in prod_indicators):
                categories['production'].append(repo_dict)
            
            # Educational
            edu_indicators = ['tutorial', 'example', 'demo', 'course', 
                            'learning', 'guide', 'workshop']
            if any(ind in description for ind in edu_indicators):
                categories['educational'].append(repo_dict)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _extract_skills(self, repositories: List[Any]) -> Dict[str, Any]:
        """
        Extract skills and technologies with proper structure.
        
        Args:
            repositories: List of repository objects
            
        Returns:
            Dictionary containing structured skill data
        """
        skills = {
            'programming_languages': {},
            'ml_frameworks': {},
            'tools_and_libraries': {},
            'techniques': set(),
            'proficiency_levels': {}
        }
        
        if not repositories:
            skills['techniques'] = []
            return skills
        
        total_repos = len(repositories)
        
        # Count programming languages
        lang_counts = {}
        for repo in repositories:
            lang = getattr(repo, 'language', None)
            if lang:
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        # Structure language data with counts and percentages
        skills['programming_languages'] = {
            lang: {
                "count": count,
                "percentage": round((count / total_repos) * 100, 1)
            }
            for lang, count in sorted(
                lang_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        }
        
        # Count ML frameworks
        framework_counts = {}
        for repo in repositories:
            frameworks = getattr(repo, 'frameworks', {})
            if not isinstance(frameworks, dict):
                continue
                
            for category, fw_list in frameworks.items():
                if not isinstance(fw_list, list):
                    continue
                    
                for framework in fw_list:
                    name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                    framework_counts[name] = framework_counts.get(name, 0) + 1
        
        skills['ml_frameworks'] = framework_counts
        
        # Extract techniques from code patterns
        for repo in repositories:
            code_analysis = getattr(repo, 'code_analysis', {})
            if not isinstance(code_analysis, dict):
                continue
                
            code_patterns = code_analysis.get('code_patterns', {})
            if isinstance(code_patterns, dict):
                for pattern, count in code_patterns.items():
                    if count > 0:
                        technique = pattern.replace('_', ' ').title()
                        skills['techniques'].add(technique)
        
        # Calculate proficiency levels
        thresholds = self.config['proficiency_thresholds']
        for framework, count in framework_counts.items():
            usage_ratio = count / total_repos
            
            if usage_ratio >= thresholds['advanced']:
                level = 'Advanced'
            elif usage_ratio >= thresholds['intermediate']:
                level = 'Intermediate'
            else:
                level = 'Beginner'
            
            skills['proficiency_levels'][framework] = {
                'level': level,
                'usage_count': count,
                'usage_ratio': round(usage_ratio, 3),
                'usage_percentage': round(usage_ratio * 100, 1)
            }
        
        # Convert techniques set to sorted list
        skills['techniques'] = sorted(list(skills['techniques']))
        
        return skills
    
    def _select_highlights(self, repositories: List[Any], 
                          max_highlights: int = 5) -> List[Dict[str, Any]]:
        """
        Select top repositories as portfolio highlights.
        
        Args:
            repositories: List of repository objects
            max_highlights: Maximum number of highlights to return
            
        Returns:
            List of highlight dictionaries
        """
        # Sort repositories by relevance score and stars
        sorted_repos = sorted(
            repositories,
            key=lambda x: (
                getattr(x, 'ai_ml_relevance_score', 0),
                getattr(x, 'stars', 0)
            ),
            reverse=True
        )
        
        highlights = []
        for repo in sorted_repos[:max_highlights]:
            highlight = {
                'name': getattr(repo, 'name', 'Unknown'),
                'full_name': getattr(repo, 'full_name', ''),
                'description': getattr(repo, 'description', ''),
                'url': getattr(repo, 'url', ''),
                'stars': getattr(repo, 'stars', 0),
                'forks': getattr(repo, 'forks', 0),
                'language': getattr(repo, 'language', ''),
                'topics': getattr(repo, 'topics', []) or [],
                'relevance_score': getattr(repo, 'ai_ml_relevance_score', 0),
                'key_frameworks': self._get_key_frameworks(
                    getattr(repo, 'frameworks', {})
                ),
                'highlights': self._generate_repo_highlights(repo)
            }
            highlights.append(highlight)
        
        return highlights
    
    def _get_key_frameworks(self, frameworks: Dict[str, List]) -> List[str]:
        """
        Extract the most important frameworks for a repository.
        
        Args:
            frameworks: Dictionary of framework categories
            
        Returns:
            List of key framework names
        """
        if not isinstance(frameworks, dict):
            return []
        
        key_frameworks = []
        
        # Priority order for framework categories
        priority_categories = [
            'llm_frameworks', 
            'ml_frameworks', 
            'deep_learning', 
            'computer_vision'
        ]
        
        for category in priority_categories:
            category_frameworks = frameworks.get(category, [])
            if not isinstance(category_frameworks, list):
                continue
                
            for framework in category_frameworks[:2]:  # Top 2 from each category
                name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                if name not in key_frameworks:
                    key_frameworks.append(name)
                
                if len(key_frameworks) >= 5:
                    break
            
            if len(key_frameworks) >= 5:
                break
        
        return key_frameworks
    
    def _generate_repo_highlights(self, repo: Any) -> List[str]:
        """
        Generate key highlights for a repository.
        
        Args:
            repo: Repository object
            
        Returns:
            List of highlight strings
        """
        highlights = []
        
        stars = getattr(repo, 'stars', 0)
        forks = getattr(repo, 'forks', 0)
        
        # Stars and forks
        if stars > 100:
            highlights.append(f"{stars} GitHub stars")
        elif stars > 50:
            highlights.append(f"{stars} stars")
            
        if forks > 20:
            highlights.append(f"{forks} forks")
        
        # Code analysis highlights
        code_analysis = getattr(repo, 'code_analysis', {})
        if isinstance(code_analysis, dict):
            if code_analysis.get('has_notebooks'):
                highlights.append("Jupyter notebook examples")
            if code_analysis.get('has_models'):
                highlights.append("Pre-trained models included")
            if code_analysis.get('has_tests'):
                highlights.append("Comprehensive test suite")
        
        # Framework diversity
        frameworks = getattr(repo, 'frameworks', {})
        if isinstance(frameworks, dict):
            total_frameworks = sum(
                len(fw_list) for fw_list in frameworks.values() 
                if isinstance(fw_list, list)
            )
            if total_frameworks >= 3:
                highlights.append("Multi-framework implementation")
        
        # AI/ML relevance
        relevance = getattr(repo, 'ai_ml_relevance_score', 0)
        if relevance >= 8.0:
            highlights.append("High AI/ML relevance")
        elif relevance >= 6.0:
            highlights.append("Strong AI/ML focus")
        
        return highlights[:4]  # Limit to top 4 highlights
    
    def _calculate_expertise_metrics(self, repositories: List[Any]) -> Dict[str, float]:
        """
        Calculate expertise metrics across different AI/ML areas.
        
        Args:
            repositories: List of repository objects
            
        Returns:
            Dictionary of expertise scores (0-10 scale)
        """
        metrics = {
            'machine_learning': 0.0,
            'deep_learning': 0.0,
            'data_science': 0.0,
            'software_engineering': 0.0,
            'research_ability': 0.0,
            'collaboration': 0.0,
            'overall_score': 0.0
        }
        
        if not repositories:
            return metrics
        
        total_repos = len(repositories)
        
        # Machine Learning expertise
        ml_repos = sum(
            1 for repo in repositories
            if any(
                fw in str(getattr(repo, 'frameworks', {}).get('ml_frameworks', []))
                for fw in ['scikit-learn', 'xgboost', 'lightgbm']
            )
        )
        metrics['machine_learning'] = min((ml_repos / total_repos) * 10, 10.0)
        
        # Deep Learning expertise
        dl_repos = sum(
            1 for repo in repositories
            if any(
                fw in str(getattr(repo, 'frameworks', {}).get('ml_frameworks', [])) +
                        str(getattr(repo, 'frameworks', {}).get('deep_learning', []))
                for fw in ['tensorflow', 'pytorch', 'keras']
            )
        )
        metrics['deep_learning'] = min((dl_repos / total_repos) * 10, 10.0)
        
        # Data Science expertise
        ds_repos = sum(
            1 for repo in repositories
            if any(
                fw in str(getattr(repo, 'frameworks', {}).get('data_frameworks', []))
                for fw in ['pandas', 'numpy']
            )
        )
        metrics['data_science'] = min((ds_repos / total_repos) * 10, 10.0)
        
        # Software Engineering (based on code quality indicators)
        se_score = 0
        for repo in repositories:
            code_analysis = getattr(repo, 'code_analysis', {})
            if isinstance(code_analysis, dict):
                if code_analysis.get('has_tests'):
                    se_score += 1
                quality_metrics = code_analysis.get('quality_metrics', {})
                if isinstance(quality_metrics, dict):
                    if quality_metrics.get('documentation', 0) > 0:
                        se_score += 0.5
        
        metrics['software_engineering'] = min((se_score / total_repos) * 10, 10.0)
        
        # Research ability
        research_score = sum(
            1 for repo in repositories
            if getattr(repo, 'code_analysis', {}).get('has_notebooks', False)
        )
        metrics['research_ability'] = min((research_score / total_repos) * 10, 10.0)
        
        # Collaboration
        total_forks = sum(getattr(repo, 'forks', 0) for repo in repositories)
        avg_forks = total_forks / total_repos
        metrics['collaboration'] = min((avg_forks / 5.0) * 10, 10.0)
        
        # Overall score (weighted average)
        weights = {
            'machine_learning': 0.25,
            'deep_learning': 0.20,
            'data_science': 0.20,
            'software_engineering': 0.15,
            'research_ability': 0.10,
            'collaboration': 0.10
        }
        
        metrics['overall_score'] = sum(
            metrics[area] * weight for area, weight in weights.items()
        )
        
        # Round all scores to 1 decimal place
        return {k: round(min(v, 10.0), 1) for k, v in metrics.items()}
    
    def generate_html_report(self, portfolio: PortfolioData) -> str:
        """
        Generate HTML portfolio report.
        
        Args:
            portfolio: PortfolioData object
            
        Returns:
            HTML string
        """
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI/ML Portfolio Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        .header {
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 3px solid #667eea;
        }
        
        .header h1 {
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #7f8c8d;
            font-size: 1.1em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 0.95em;
            opacity: 0.95;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .section {
            margin: 50px 0;
            padding: 30px;
            border-left: 5px solid #667eea;
            background: linear-gradient(to right, #f8f9fa 0%, #ffffff 100%);
            border-radius: 8px;
        }
        
        .section h2 {
            color: #2c3e50;
            margin-bottom: 25px;
            font-size: 1.8em;
        }
        
        .highlight-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .highlight-card:hover {
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
        
        .highlight-title {
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.3em;
            display: block;
            margin-bottom: 10px;
        }
        
        .highlight-title:hover {
            color: #764ba2;
        }
        
        .highlight-meta {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 10px 0;
        }
        
        .tags {
            margin: 15px 0;
        }
        
        .tag {
            display: inline-block;
            background: #e74c3c;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 4px;
            font-weight: 500;
        }
        
        .framework-tag {
            background: #27ae60;
        }
        
        .skill-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin: 25px 0;
        }
        
        .skill-category {
            background: white;
            border-radius: 10px;
            padding: 25px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .skill-category h3 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.2em;
        }
        
        .skill-item {
            margin: 15px 0;
        }
        
        .skill-name {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .skill-bar {
            background: #ecf0f1;
            border-radius: 10px;
            height: 22px;
            overflow: hidden;
            position: relative;
        }
        
        .skill-progress {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 8px;
            color: white;
            font-size: 0.75em;
            font-weight: bold;
        }
        
        .skill-details {
            font-size: 0.85em;
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        .expertise-radar {
            text-align: center;
            margin: 30px 0;
        }
        
        .chart-placeholder {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border: 2px dashed #95a5a6;
            height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #34495e;
            border-radius: 12px;
            padding: 30px;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .metric-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .category-section {
            margin: 30px 0;
        }
        
        .category-header {
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
        }
        
        .repo-item {
            margin: 12px 0;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            transition: all 0.2s ease;
        }
        
        .repo-item:hover {
            border-color: #667eea;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        
        .repo-item a {
            color: #667eea;
            text-decoration: none;
            font-weight: 600;
        }
        
        .repo-item a:hover {
            color: #764ba2;
        }
        
        .repo-description {
            color: #7f8c8d;
            font-size: 0.95em;
            margin-left: 10px;
        }
        
        .timeline-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        
        .timeline-year {
            font-weight: 600;
            color: #2c3e50;
        }
        
        .timeline-count {
            color: #667eea;
            font-weight: 500;
        }
        
        .framework-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .framework-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .framework-item:hover {
            border-color: #667eea;
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        
        .framework-name {
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 8px;
        }
        
        .framework-count {
            color: #667eea;
            font-size: 1.4em;
            font-weight: bold;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            .container {
                box-shadow: none;
                padding: 20px;
            }
            .stat-card, .highlight-card {
                break-inside: avoid;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            .header h1 {
                font-size: 1.8em;
            }
            .stats-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI/ML Portfolio Report</h1>
            <p>Generated on {{ portfolio.generation_date[:10] }}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ portfolio.total_repositories }}</div>
                <div class="stat-label">Repositories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ portfolio.insights.total_stars }}</div>
                <div class="stat-label">GitHub Stars</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ portfolio.insights.total_forks }}</div>
                <div class="stat-label">Total Forks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ portfolio.expertise_metrics.overall_score }}</div>
                <div class="stat-label">Overall Score</div>
            </div>
        </div>
        
        <div class="section">
            <h2>🌟 Portfolio Highlights</h2>
            {% for highlight in portfolio.highlights %}
            <div class="highlight-card">
                <a href="{{ highlight.url }}" class="highlight-title" target="_blank">
                    {{ highlight.name }}
                </a>
                <div class="highlight-meta">
                    ⭐ {{ highlight.stars }} stars • 🍴 {{ highlight.forks }} forks • 
                    📊 {{ highlight.relevance_score|round(1) }}/10 relevance
                </div>
                <p style="margin: 15px 0; line-height: 1.6;">{{ highlight.description }}</p>
                <div class="tags">
                    {% for framework in highlight.key_frameworks %}
                    <span class="tag framework-tag">{{ framework }}</span>
                    {% endfor %}
                    {% for topic in highlight.topics[:3] %}
                    <span class="tag">{{ topic }}</span>
                    {% endfor %}
                </div>
                {% if highlight.highlights %}
                <ul style="margin-top: 15px; padding-left: 20px;">
                    {% for point in highlight.highlights %}
                    <li style="margin: 5px 0;">{{ point }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>🛠️ Technical Skills</h2>
            <div class="skill-grid">
                <div class="skill-category">
                    <h3>Programming Languages</h3>
                    {% for lang, data in portfolio.skills.programming_languages.items() %}
                    <div class="skill-item">
                        <div class="skill-name">
                            <span>{{ lang }}</span>
                            <span>{{ data.percentage }}%</span>
                        </div>
                        <div class="skill-bar">
                            <div class="skill-progress" style="width: {{ data.percentage }}%">
                            </div>
                        </div>
                        <div class="skill-details">{{ data.count }} repositories</div>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="skill-category">
                    <h3>ML/AI Frameworks</h3>
                    {% for framework, count in portfolio.skills.ml_frameworks.items() %}
                    {% if loop.index <= 10 %}
                    <div class="skill-item">
                        <div class="skill-name">
                            <span>{{ framework }}</span>
                            <span>{{ portfolio.skills.proficiency_levels.get(framework, {}).get('usage_percentage', 0)|round(1) }}%</span>
                        </div>
                        <div class="skill-bar">
                            <div class="skill-progress" style="width: {{ (count / portfolio.total_repositories * 100) }}%">
                            </div>
                        </div>
                        <div class="skill-details">
                            {{ count }} projects • 
                            {{ portfolio.skills.proficiency_levels.get(framework, {}).get('level', 'Beginner') }}
                        </div>
                    </div>
                    {% endif %}
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Expertise Metrics</h2>
            <div class="expertise-radar">
                <div class="chart-placeholder">
                    <h3 style="margin-bottom: 20px;">Expertise Breakdown</h3>
                    <div class="metric-grid">
                        <div class="metric-item">
                            <div class="metric-label">Machine Learning</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.machine_learning }}/10</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Deep Learning</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.deep_learning }}/10</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Data Science</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.data_science }}/10</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Software Engineering</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.software_engineering }}/10</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Research</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.research_ability }}/10</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Collaboration</div>
                            <div class="metric-value">{{ portfolio.expertise_metrics.collaboration }}/10</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📂 Repository Categories</h2>
            {% for category, repos in portfolio.categories.items() %}
            <div class="category-section">
                <div class="category-header">
                    <strong>{{ category.replace('_', ' ').title() }}</strong> 
                    ({{ repos|length }} repositories)
                </div>
                <div>
                    {% for repo in repos[:3] %}
                    <div class="repo-item">
                        <strong><a href="{{ repo.url }}" target="_blank">{{ repo.name }}</a></strong>
                        <span class="repo-description">
                            {% if repo.description %}
                            - {{ repo.description[:100] }}{% if repo.description|length > 100 %}...{% endif %}
                            {% endif %}
                        </span>
                    </div>
                    {% endfor %}
                    {% if repos|length > 3 %}
                    <p style="margin: 15px 0; color: #7f8c8d; font-style: italic;">
                        ... and {{ repos|length - 3 }} more repositories
                    </p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>📅 Activity Timeline</h2>
            {% for activity in portfolio.insights.activity_timeline %}
            <div class="timeline-item">
                <span class="timeline-year">{{ activity.year }}</span>
                <span class="timeline-count">{{ activity.repositories }} repositories</span>
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>🔍 Framework Usage</h2>
            <div class="framework-grid">
                {% for framework, count in portfolio.insights.frameworks_summary.items() %}
                <div class="framework-item">
                    <div class="framework-name">{{ framework }}</div>
                    <div class="framework-count">{{ count }}</div>
                    <div style="font-size: 0.85em; color: #7f8c8d; margin-top: 5px;">
                        projects
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <script>
        // Animate skill bars on load
        document.addEventListener('DOMContentLoaded', function() {
            const skillBars = document.querySelectorAll('.skill-progress');
            skillBars.forEach(bar => {
                const width = bar.style.width;
                bar.style.width = '0%';
                setTimeout(() => {
                    bar.style.width = width;
                }, 200);
            });
        });
    </script>
</body>
</html>
        """
        
        try:
            template = Template(html_template)
            return template.render(portfolio=portfolio)
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}", exc_info=True)
            raise
    
    def generate_markdown_summary(self, portfolio: PortfolioData) -> str:
        """
        Generate markdown summary of the portfolio.
        
        Args:
            portfolio: PortfolioData object
            
        Returns:
            Markdown formatted string
        """
        try:
            md_content = f"""# 🤖 AI/ML Portfolio Summary

*Generated on {portfolio.generation_date[:10]}*

## 📊 Overview

- **Total Repositories:** {portfolio.total_repositories}
- **GitHub Stars:** {portfolio.insights.get('total_stars', 0)}
- **Total Forks:** {portfolio.insights.get('total_forks', 0)}
- **Overall Expertise Score:** {portfolio.expertise_metrics.get('overall_score', 0):.1f}/10
- **Average Stars per Repo:** {portfolio.insights.get('avg_stars', 0):.1f}
- **Average Forks per Repo:** {portfolio.insights.get('avg_forks', 0):.1f}

## 🌟 Portfolio Highlights

"""
            
            for i, highlight in enumerate(portfolio.highlights[:3], 1):
                md_content += f"""### {i}. [{highlight['name']}]({highlight['url']})

⭐ {highlight['stars']} stars • 🍴 {highlight['forks']} forks • 📊 {highlight['relevance_score']:.1f}/10 relevance

{highlight['description']}

**Key Frameworks:** {', '.join(highlight['key_frameworks']) if highlight['key_frameworks'] else 'N/A'}

**Highlights:**
"""
                for point in highlight['highlights']:
                    md_content += f"- {point}\n"
                
                md_content += "\n---\n\n"
            
            md_content += """## 🛠️ Technical Skills

### Programming Languages

"""
            
            for lang, data in list(portfolio.skills['programming_languages'].items())[:5]:
                md_content += f"- **{lang}**: {data['count']} repositories ({data['percentage']:.1f}%)\n"
            
            md_content += "\n### ML/AI Frameworks\n\n"
            
            for framework, count in list(portfolio.skills['ml_frameworks'].items())[:10]:
                proficiency_data = portfolio.skills['proficiency_levels'].get(framework, {})
                proficiency = proficiency_data.get('level', 'Beginner') if isinstance(proficiency_data, dict) else 'Beginner'
                md_content += f"- **{framework}**: {count} projects ({proficiency})\n"
            
            md_content += f"""

### Key Techniques

{', '.join(portfolio.skills.get('techniques', [])[:15]) if portfolio.skills.get('techniques') else 'N/A'}

## 📈 Expertise Breakdown

| Area | Score |
|------|-------|
| Machine Learning | {portfolio.expertise_metrics.get('machine_learning', 0):.1f}/10 |
| Deep Learning | {portfolio.expertise_metrics.get('deep_learning', 0):.1f}/10 |
| Data Science | {portfolio.expertise_metrics.get('data_science', 0):.1f}/10 |
| Software Engineering | {portfolio.expertise_metrics.get('software_engineering', 0):.1f}/10 |
| Research Ability | {portfolio.expertise_metrics.get('research_ability', 0):.1f}/10 |
| Collaboration | {portfolio.expertise_metrics.get('collaboration', 0):.1f}/10 |

## 📂 Repository Categories

"""
            
            for category, repos in portfolio.categories.items():
                category_name = category.replace('_', ' ').title()
                md_content += f"### {category_name} ({len(repos)} repositories)\n\n"
                
                for repo in repos[:3]:
                    desc = repo.get('description', '')[:100]
                    if len(repo.get('description', '')) > 100:
                        desc += '...'
                    md_content += f"- **[{repo['name']}]({repo['url']})** - {desc}\n"
                
                if len(repos) > 3:
                    md_content += f"- *... and {len(repos) - 3} more repositories*\n"
                
                md_content += "\n"
            
            md_content += """## 🚀 Key Achievements

"""
            
            # Calculate achievements
            total_stars = portfolio.insights.get('total_stars', 0)
            total_forks = portfolio.insights.get('total_forks', 0)
            
            if total_stars > 500:
                md_content += f"- 🌟 Accumulated over {total_stars} GitHub stars across projects\n"
            elif total_stars > 100:
                md_content += f"- 🌟 Earned {total_stars} GitHub stars from the community\n"
            
            if total_forks > 100:
                md_content += f"- 🍴 Projects forked {total_forks} times by the community\n"
            elif total_forks > 20:
                md_content += f"- 🍴 {total_forks} community forks across repositories\n"
            
            # Framework diversity
            unique_frameworks = len(portfolio.skills.get('ml_frameworks', {}))
            if unique_frameworks >= 5:
                md_content += f"- 🛠️ Experience with {unique_frameworks} different ML/AI frameworks\n"
            
            # Multi-category expertise
            categories_with_projects = len([
                cat for cat, repos in portfolio.categories.items() 
                if len(repos) >= 2
            ])
            if categories_with_projects >= 3:
                md_content += f"- 🎯 Expertise across {categories_with_projects} different AI/ML domains\n"
            
            # Overall score highlight
            overall_score = portfolio.expertise_metrics.get('overall_score', 0)
            if overall_score >= 7.0:
                md_content += f"- 📊 Strong overall expertise score of {overall_score:.1f}/10\n"
            
            md_content += "\n## 📅 Activity Timeline\n\n"
            
            for activity in portfolio.insights.get('activity_timeline', []):
                md_content += f"- **{activity['year']}**: {activity['repositories']} repositories\n"
            
            md_content += f"""

## 💡 Summary

This portfolio demonstrates {"strong" if overall_score >= 7.0 else "solid"} AI/ML expertise across multiple domains with {portfolio.total_repositories} analyzed repositories. The work spans {len(portfolio.categories)} different categories with particular strength in {self._get_top_category(portfolio)}.

---

*This portfolio summary was automatically generated from repository analysis.*
"""
            
            return md_content
            
        except Exception as e:
            self.logger.error(f"Error generating markdown summary: {e}", exc_info=True)
            raise
    
    def _get_top_category(self, portfolio: PortfolioData) -> str:
        """Get the category with the most repositories."""
        if not portfolio.categories:
            return "various AI/ML domains"
        
        top_category = max(
            portfolio.categories.items(),
            key=lambda x: len(x[1])
        )
        return top_category[0].replace('_', ' ').title()
    
    def export_json(self, portfolio: PortfolioData, filepath: str):
        """
        Export portfolio data as JSON.
        
        Args:
            portfolio: PortfolioData object
            filepath: Output file path
        """
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(portfolio.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Portfolio exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting JSON: {e}", exc_info=True)
            raise
    
    def export_portfolio(self, portfolio: PortfolioData, 
                        output_dir: str = "./portfolio_output",
                        formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export portfolio in multiple formats.
        
        Args:
            portfolio: Portfolio data to export
            output_dir: Output directory path
            formats: List of formats ['html', 'markdown', 'json']. 
                    If None, exports all formats.
        
        Returns:
            Dictionary mapping format to output file path
            
        Raises:
            ValueError: If invalid format specified
        """
        if formats is None:
            formats = ['html', 'markdown', 'json']
        
        valid_formats = {'html', 'markdown', 'json'}
        invalid = set(formats) - valid_formats
        if invalid:
            raise ValueError(f"Invalid formats: {invalid}. Must be one of {valid_formats}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if 'html' in formats:
                html_content = self.generate_html_report(portfolio)
                html_file = output_path / f"portfolio_report_{timestamp}.html"
                html_file.write_text(html_content, encoding='utf-8')
                outputs['html'] = str(html_file)
                self.logger.info(f"HTML report exported to {html_file}")
            
            if 'markdown' in formats:
                md_content = self.generate_markdown_summary(portfolio)
                md_file = output_path / f"portfolio_summary_{timestamp}.md"
                md_file.write_text(md_content, encoding='utf-8')
                outputs['markdown'] = str(md_file)
                self.logger.info(f"Markdown summary exported to {md_file}")
            
            if 'json' in formats:
                json_file = output_path / f"portfolio_data_{timestamp}.json"
                self.export_json(portfolio, str(json_file))
                outputs['json'] = str(json_file)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error exporting portfolio: {e}", exc_info=True)
            raise
    
    def generate_skills_matrix(self, portfolio: PortfolioData) -> Dict[str, Any]:
        """
        Generate a detailed skills matrix.
        
        Args:
            portfolio: PortfolioData object
            
        Returns:
            Dictionary containing detailed skills analysis
        """
        matrix = {
            'technical_skills': {},
            'domain_expertise': {},
            'proficiency_levels': {},
            'growth_areas': [],
            'strengths': []
        }
        
        total_repos = portfolio.total_repositories
        if total_repos == 0:
            return matrix
        
        # Technical skills matrix
        frameworks = portfolio.skills.get('ml_frameworks', {})
        
        for framework, count in frameworks.items():
            usage_ratio = count / total_repos
            proficiency_data = portfolio.skills.get('proficiency_levels', {}).get(framework, {})
            
            matrix['technical_skills'][framework] = {
                'usage_count': count,
                'usage_percentage': round(usage_ratio * 100, 1),
                'proficiency_level': proficiency_data.get('level', 'Beginner') if isinstance(proficiency_data, dict) else 'Beginner',
                'experience_level': self._calculate_experience_level(usage_ratio)
            }
        
        # Domain expertise
        for category, repos in portfolio.categories.items():
            if repos:
                expertise_score = (len(repos) / total_repos) * 10
                matrix['domain_expertise'][category] = {
                    'project_count': len(repos),
                    'expertise_score': round(expertise_score, 1),
                    'key_projects': [repo['name'] for repo in repos[:3]]
                }
        
        # Identify strengths (high expertise areas)
        for category, data in matrix['domain_expertise'].items():
            if data['expertise_score'] >= 3.0:
                matrix['strengths'].append({
                    'domain': category,
                    'score': data['expertise_score'],
                    'project_count': data['project_count']
                })
        
        # Sort strengths by score
        matrix['strengths'].sort(key=lambda x: x['score'], reverse=True)
        
        # Identify growth areas
        all_categories = [
            'machine_learning', 'deep_learning', 'natural_language_processing', 
            'computer_vision', 'data_science', 'research', 'production'
        ]
        
        for category in all_categories:
            if category not in portfolio.categories or len(portfolio.categories[category]) < 2:
                matrix['growth_areas'].append(category)
        
        matrix['proficiency_levels'] = portfolio.skills.get('proficiency_levels', {})
        
        return matrix
    
    def _calculate_experience_level(self, usage_ratio: float) -> str:
        """
        Calculate experience level based on usage ratio.
        
        Args:
            usage_ratio: Ratio of repositories using the skill
            
        Returns:
            Experience level string
        """
        if usage_ratio >= 0.4:
            return 'Expert'
        elif usage_ratio >= 0.2:
            return 'Proficient'
        elif usage_ratio >= 0.1:
            return 'Competent'
        else:
            return 'Novice'
    
    def generate_statistics_summary(self, portfolio: PortfolioData) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary.
        
        Args:
            portfolio: PortfolioData object
            
        Returns:
            Dictionary containing statistical analysis
        """
        stats = {
            'overview': {
                'total_repositories': portfolio.total_repositories,
                'total_stars': portfolio.insights.get('total_stars', 0),
                'total_forks': portfolio.insights.get('total_forks', 0),
                'average_stars': portfolio.insights.get('avg_stars', 0),
                'average_forks': portfolio.insights.get('avg_forks', 0)
            },
            'technology_diversity': {
                'unique_languages': len(portfolio.skills.get('programming_languages', {})),
                'unique_frameworks': len(portfolio.skills.get('ml_frameworks', {})),
                'unique_techniques': len(portfolio.skills.get('techniques', []))
            },
            'domain_coverage': {
                category: len(repos) 
                for category, repos in portfolio.categories.items()
            },
            'expertise_summary': portfolio.expertise_metrics,
            'top_skills': self._get_top_skills(portfolio, top_n=10),
            'collaboration_metrics': {
                'avg_forks': portfolio.insights.get('avg_forks', 0),
                'collaboration_score': portfolio.insights.get('collaboration_score', 0),
                'highly_forked_repos': len([
                    r for r in portfolio.repositories 
                    if getattr(r, 'forks', 0) > 10
                ]),
                'starred_repos': len([
                    r for r in portfolio.repositories 
                    if getattr(r, 'stars', 0) > 50
                ])
            },
            'quality_metrics': {
                'code_quality_score': portfolio.insights.get('code_quality_score', 0),
                'repos_with_tests': len([
                    r for r in portfolio.repositories
                    if getattr(r, 'code_analysis', {}).get('has_tests', False)
                ]),
                'repos_with_notebooks': len([
                    r for r in portfolio.repositories
                    if getattr(r, 'code_analysis', {}).get('has_notebooks', False)
                ])
            }
        }
        
        return stats
    
    def _get_top_skills(self, portfolio: PortfolioData, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract top skills with usage statistics.
        
        Args:
            portfolio: PortfolioData object
            top_n: Number of top skills to return
            
        Returns:
            List of skill dictionaries
        """
        skills = []
        
        # Get frameworks with counts
        for framework, count in portfolio.skills.get('ml_frameworks', {}).items():
            proficiency_data = portfolio.skills.get('proficiency_levels', {}).get(framework, {})
            proficiency = proficiency_data.get('level', 'Beginner') if isinstance(proficiency_data, dict) else 'Beginner'
            
            skills.append({
                'name': framework,
                'type': 'framework',
                'usage_count': count,
                'proficiency': proficiency,
                'usage_percentage': proficiency_data.get('usage_percentage', 0) if isinstance(proficiency_data, dict) else 0
            })
        
        # Sort by usage count
        skills.sort(key=lambda x: x['usage_count'], reverse=True)
        
        return skills[:top_n]
    
    def compare_portfolios(self, portfolio1: PortfolioData, 
                          portfolio2: PortfolioData) -> Dict[str, Any]:
        """
        Compare two portfolios and highlight differences.
        
        Useful for tracking growth over time or comparing profiles.
        
        Args:
            portfolio1: First portfolio (typically older)
            portfolio2: Second portfolio (typically newer)
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {
            'repository_growth': portfolio2.total_repositories - portfolio1.total_repositories,
            'star_growth': (
                portfolio2.insights.get('total_stars', 0) - 
                portfolio1.insights.get('total_stars', 0)
            ),
            'fork_growth': (
                portfolio2.insights.get('total_forks', 0) - 
                portfolio1.insights.get('total_forks', 0)
            ),
            'new_skills': [],
            'improved_areas': [],
            'expertise_changes': {},
            'new_categories': [],
            'growth_percentage': 0.0
        }
        
        # Calculate growth percentage
        if portfolio1.total_repositories > 0:
            comparison['growth_percentage'] = round(
                (comparison['repository_growth'] / portfolio1.total_repositories) * 100,
                1
            )
        
        # Find new frameworks
        old_frameworks = set(portfolio1.skills.get('ml_frameworks', {}).keys())
        new_frameworks = set(portfolio2.skills.get('ml_frameworks', {}).keys())
        comparison['new_skills'] = sorted(list(new_frameworks - old_frameworks))
        
        # Find new categories
        old_categories = set(portfolio1.categories.keys())
        new_categories = set(portfolio2.categories.keys())
        comparison['new_categories'] = sorted(list(new_categories - old_categories))
        
        # Compare expertise metrics
        for area, score2 in portfolio2.expertise_metrics.items():
            score1 = portfolio1.expertise_metrics.get(area, 0)
            change = round(score2 - score1, 1)
            
            if abs(change) > 0.5:  # Significant change
                comparison['expertise_changes'][area] = {
                    'old_score': score1,
                    'new_score': score2,
                    'change': change,
                    'improved': change > 0,
                    'change_percentage': round((change / max(score1, 0.1)) * 100, 1) if score1 > 0 else 0
                }
                
                if change > 0:
                    comparison['improved_areas'].append(area)
        
        return comparison
    
    def generate_recommendations(self, portfolio: PortfolioData) -> Dict[str, List[str]]:
        """
        Generate recommendations for portfolio improvement.
        
        Args:
            portfolio: PortfolioData object
            
        Returns:
            Dictionary containing categorized recommendations
        """
        recommendations = {
            'skills_to_learn': [],
            'categories_to_explore': [],
            'best_practices': [],
            'collaboration_tips': []
        }
        
        total_repos = portfolio.total_repositories
        if total_repos == 0:
            recommendations['best_practices'].append("Start building AI/ML projects to showcase your skills")
            return recommendations
        
        # Analyze current state
        frameworks = portfolio.skills.get('ml_frameworks', {})
        categories = portfolio.categories
        expertise = portfolio.expertise_metrics
        
        # Skills recommendations
        common_frameworks = {
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 
            'numpy', 'transformers', 'keras'
        }
        current_frameworks = set(fw.lower() for fw in frameworks.keys())
        missing_frameworks = common_frameworks - current_frameworks
        
        if missing_frameworks:
            recommendations['skills_to_learn'].extend([
                f"Consider learning {fw.title()} to broaden your ML toolkit"
                for fw in list(missing_frameworks)[:3]
            ])
        
        # Category recommendations
        weak_categories = [
            cat for cat, score in expertise.items()
            if score < 3.0 and cat not in ['collaboration', 'overall_score']
        ]
        
        if weak_categories:
            for cat in weak_categories[:2]:
                readable_cat = cat.replace('_', ' ').title()
                recommendations['categories_to_explore'].append(
                    f"Expand your {readable_cat} portfolio with more projects"
                )
        
        # Best practices
        repos_with_tests = sum(
            1 for r in portfolio.repositories
            if getattr(r, 'code_analysis', {}).get('has_tests', False)
        )
        
        if repos_with_tests < total_repos * 0.5:
            recommendations['best_practices'].append(
                "Add comprehensive tests to more projects to demonstrate software engineering skills"
            )
        
        repos_with_docs = sum(
            1 for r in portfolio.repositories
            if getattr(r, 'code_analysis', {}).get('quality_metrics', {}).get('documentation', 0) > 0
        )
        
        if repos_with_docs < total_repos * 0.7:
            recommendations['best_practices'].append(
                "Improve documentation in your projects for better accessibility"
            )
        
        avg_stars = portfolio.insights.get('avg_stars', 0)
        if avg_stars < 10:
            recommendations['best_practices'].append(
                "Promote your projects to gain more visibility and community engagement"
            )
        
        # Collaboration tips
        avg_forks = portfolio.insights.get('avg_forks', 0)
        if avg_forks < 2:
            recommendations['collaboration_tips'].append(
                "Make your repositories more fork-friendly with clear CONTRIBUTING.md files"
            )
            recommendations['collaboration_tips'].append(
                "Add issues and encourage community contributions"
            )
        
        if not any('production' in cat for cat in categories.keys()):
            recommendations['categories_to_explore'].append(
                "Deploy some projects to production to demonstrate end-to-end ML skills"
            )
        
        return recommendations


# Convenience function for quick portfolio generation
def build_and_export_portfolio(repositories: List[Any],
                               output_dir: str = "./portfolio_output",
                               config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Convenience function to build and export portfolio in one call.
    
    Args:
        repositories: List of analyzed repository objects
        output_dir: Output directory for exported files
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping format to output file path
        
    Example:
        >>> repos = analyzer.analyze_repositories(repo_list)
        >>> outputs = build_and_export_portfolio(repos)
        >>> print(f"Portfolio generated: {outputs}")
    """
    builder = PortfolioBuilder(config)
    portfolio = builder.build_portfolio(repositories, config)
    return builder.export_portfolio(portfolio, output_dir)


if __name__ == "__main__":
    # Example usage and testing
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Portfolio Builder Module")
    print("=" * 50)
    print("\nThis module provides comprehensive AI/ML portfolio building capabilities.")
    print("\nKey Features:")
    print("  • Multi-format export (HTML, Markdown, JSON)")
    print("  • Detailed skill extraction and categorization")
    print("  • Expertise metrics calculation")
    print("  • Portfolio comparison and recommendations")
    print("\nUsage:")
    print("  builder = PortfolioBuilder()")
    print("  portfolio = builder.build_portfolio(repositories)")
    print("  outputs = builder.export_portfolio(portfolio)")
    print("\nFor more information, see the docstrings and examples.")
    print("=" * 50)