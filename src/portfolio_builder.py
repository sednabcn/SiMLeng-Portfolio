#!/usr/bin/env python3
"""Portfolio Builder with safe template rendering"""
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from jinja2 import Template

class PortfolioBuilder:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_portfolio(self, repositories: List[Any], config: Dict[str, Any]):
        from data_models import PortfolioData
        portfolio = PortfolioData()
        portfolio.total_repositories = len(repositories)
        portfolio.generation_date = datetime.now().isoformat()
        portfolio.repositories = [r for r in repositories if r.ai_ml_relevance_score >= 2.0]
        
        portfolio.insights = {
            'total_stars': sum(r.stars for r in portfolio.repositories),
            'total_forks': sum(r.forks for r in portfolio.repositories),
            'languages': {},
            'frameworks_summary': {},
            'activity_timeline': []
        }
        
        # Languages
        for repo in portfolio.repositories:
            if repo.language:
                lang = repo.language
                if lang not in portfolio.insights['languages']:
                    portfolio.insights['languages'][lang] = {'count': 0, 'percentage': 0}
                portfolio.insights['languages'][lang]['count'] += 1
        
        # Calculate percentages
        total = len(portfolio.repositories) or 1
        for lang in portfolio.insights['languages']:
            portfolio.insights['languages'][lang]['percentage'] = (
                portfolio.insights['languages'][lang]['count'] / total * 100
            )
        
        portfolio.categories = {}
        portfolio.skills = {
            'programming_languages': portfolio.insights['languages'],
            'ml_frameworks': {},
            'proficiency_levels': {}
        }
        
        portfolio.highlights = []
        for repo in sorted(portfolio.repositories, key=lambda x: x.stars, reverse=True)[:5]:
            portfolio.highlights.append({
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description or 'No description',
                'url': repo.url,
                'stars': repo.stars,
                'forks': repo.forks,
                'relevance_score': repo.ai_ml_relevance_score,
                'key_frameworks': [],
                'topics': repo.topics[:3] if repo.topics else [],
                'highlights': []
            })
        
        portfolio.expertise_metrics = {
            'machine_learning': 5.0,
            'deep_learning': 4.0,
            'data_science': 6.0,
            'software_engineering': 5.0,
            'research_ability': 4.0,
            'collaboration': 3.0,
            'overall_score': 5.0
        }
        
        return portfolio
    
    def generate_html_report(self, portfolio) -> str:
        template = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AI/ML Portfolio</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
        h1 { color: #2c3e50; text-align: center; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 30px 0; }
        .stat { background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px; border-radius: 10px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; }
        .highlight { background: #f8f9fa; border: 1px solid #ddd; padding: 20px; margin: 15px 0; border-radius: 8px; }
        .tag { background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI/ML Portfolio Report</h1>
        <p style="text-align: center; color: #7f8c8d;">Generated: {{ date }}</p>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{{ total_repos }}</div>
                <div>Repositories</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ stars }}</div>
                <div>GitHub Stars</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ forks }}</div>
                <div>Forks</div>
            </div>
            <div class="stat">
                <div class="stat-value">{{ score }}</div>
                <div>Score</div>
            </div>
        </div>
        
        <h2>Top Projects</h2>
        {% for h in highlights %}
        <div class="highlight">
            <h3><a href="{{ h.url }}">{{ h.name }}</a></h3>
            <p>{{ h.description }}</p>
            <div>
                <span class="tag">⭐ {{ h.stars }}</span>
                <span class="tag">🍴 {{ h.forks }}</span>
                <span class="tag">Score: {{ h.relevance_score|round(1) }}/10</span>
            </div>
        </div>
        {% endfor %}
        
        <h2>Languages</h2>
        {% for lang, data in languages.items() %}
        <div style="margin: 10px 0;">
            <strong>{{ lang }}</strong>: {{ data.count }} projects ({{ data.percentage|round(0) }}%)
        </div>
        {% endfor %}
    </div>
</body>
</html>""")
        
        return template.render(
            date=portfolio.generation_date[:10],
            total_repos=portfolio.total_repositories,
            stars=portfolio.insights['total_stars'],
            forks=portfolio.insights['total_forks'],
            score=round(portfolio.expertise_metrics['overall_score'], 1),
            highlights=portfolio.highlights,
            languages=portfolio.insights['languages']
        )
    
    def generate_markdown_summary(self, portfolio) -> str:
        md = f"""# AI/ML Portfolio Summary

Generated: {portfolio.generation_date[:10]}

## Overview
- Repositories: {portfolio.total_repositories}
- Stars: {portfolio.insights['total_stars']}
- Forks: {portfolio.insights['total_forks']}

## Top Projects

"""
        for h in portfolio.highlights[:3]:
            md += f"### [{h['name']}]({h['url']})\n{h['description']}\n\n"
        
        return md

class PortfolioData:
    def __init__(self):
        self.total_repositories = 0
        self.generation_date = ""
        self.repositories = []
        self.insights = {}
        self.categories = {}
        self.skills = {}
        self.highlights = []
        self.expertise_metrics = {}
    
    def to_dict(self):
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
