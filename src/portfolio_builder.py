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
        
        # Sort by relevance score (highest first) and keep repos with score >= 1.0
        sorted_repos = sorted(repositories, key=lambda r: r.ai_ml_relevance_score, reverse=True)
        portfolio.repositories = [r for r in sorted_repos if r.ai_ml_relevance_score >= 1.0]
        
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
        
        # Top 5 projects by score (not just stars!)
        portfolio.highlights = []
        for repo in portfolio.repositories[:5]:  # Already sorted by score
            portfolio.highlights.append({
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description or 'No description',
                'url': repo.url,
                'stars': repo.stars,
                'forks': repo.forks,
                'language': repo.language,
                'relevance_score': repo.ai_ml_relevance_score,
                'key_frameworks': [],
                'topics': repo.topics[:5] if repo.topics else [],
                'highlights': []
            })
        
        # Calculate average score
        if portfolio.repositories:
            avg_score = sum(r.ai_ml_relevance_score for r in portfolio.repositories) / len(portfolio.repositories)
        else:
            avg_score = 0.0
        
        portfolio.expertise_metrics = {
            'machine_learning': 5.0,
            'deep_learning': 4.0,
            'data_science': 6.0,
            'software_engineering': 5.0,
            'research_ability': 4.0,
            'collaboration': 3.0,
            'overall_score': avg_score
        }
        
        return portfolio
    
    def generate_html_report(self, portfolio) -> str:
        """Generate improved HTML with expandable sections and better styling"""
        
        # Separate top 5 from the rest
        top_5 = portfolio.highlights
        all_repos = portfolio.repositories
        other_repos = all_repos[5:] if len(all_repos) > 5 else []
        
        template = Template("""<!DOCTYPE html>
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
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .generated-date {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-number {
            font-size: 3em;
            font-weight: bold;
            display: block;
            margin-bottom: 10px;
        }
        
        .stat-label {
            font-size: 1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        h2 {
            font-size: 2em;
            margin-bottom: 30px;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .project-grid {
            display: grid;
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .project-card {
            border: 2px solid #e0e0e0;
            border-radius: 15px;
            padding: 25px;
            background: white;
            transition: all 0.3s ease;
        }
        
        .project-card:hover {
            border-color: #667eea;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        
        .project-card h3 {
            color: #667eea;
            font-size: 1.5em;
            margin-bottom: 15px;
        }
        
        .project-card h3 a {
            color: #667eea;
            text-decoration: none;
        }
        
        .project-card h3 a:hover {
            text-decoration: underline;
        }
        
        .project-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        
        .project-meta {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .badge-stars {
            background: #ffd700;
            color: #333;
        }
        
        .badge-forks {
            background: #4CAF50;
            color: white;
        }
        
        .badge-score {
            background: #667eea;
            color: white;
        }
        
        .badge-language {
            background: #e0e0e0;
            color: #333;
        }
        
        .topics {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        
        .topic-tag {
            background: #f0f0f0;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            color: #555;
        }
        
        .expandable {
            margin-bottom: 30px;
        }
        
        .expand-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            text-align: center;
        }
        
        .expand-button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .expand-button:active {
            transform: scale(0.98);
        }
        
        .expandable-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.5s ease;
        }
        
        .expandable-content.expanded {
            max-height: 20000px;
        }
        
        .languages {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .language-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .language-name {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 5px;
        }
        
        .language-stats {
            color: #666;
            font-size: 0.9em;
        }
        
        footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }
        
        @media (max-width: 768px) {
            .stats {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.8em;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🤖 AI/ML Portfolio Report</h1>
            <p class="generated-date">Generated: {{ date }}</p>
        </header>
        
        <div class="stats">
            <div class="stat-card" onclick="scrollToProjects()">
                <span class="stat-number">{{ total_repos }}</span>
                <span class="stat-label">Repositories</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{{ stars }}</span>
                <span class="stat-label">GitHub Stars</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{{ forks }}</span>
                <span class="stat-label">Forks</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{{ score }}</span>
                <span class="stat-label">Score</span>
            </div>
        </div>
        
        <div class="content" id="projects">
            <h2>🏆 Top Projects</h2>
            <div class="project-grid">
                {% for h in highlights %}
                <div class="project-card">
                    <h3><a href="{{ h.url }}" target="_blank">{{ h.name }}</a></h3>
                    <p class="project-description">{{ h.description }}</p>
                    <div class="project-meta">
                        <span class="badge badge-stars">⭐ {{ h.stars }}</span>
                        <span class="badge badge-forks">🔱 {{ h.forks }}</span>
                        <span class="badge badge-score">Score: {{ h.relevance_score|round(1) }}/10</span>
                        {% if h.language %}
                        <span class="badge badge-language">{{ h.language }}</span>
                        {% endif %}
                    </div>
                    {% if h.topics %}
                    <div class="topics">
                        {% for topic in h.topics %}
                        <span class="topic-tag">{{ topic }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            {% if other_repos %}
            <div class="expandable">
                <button class="expand-button" onclick="toggleExpand()" id="expandBtn">
                    📁 Show All {{ other_count }} More Projects
                </button>
                <div class="expandable-content" id="allProjects">
                    <div class="project-grid" style="margin-top: 20px;">
                        {% for repo in other_repos %}
                        <div class="project-card">
                            <h3><a href="{{ repo.url }}" target="_blank">{{ repo.name }}</a></h3>
                            <p class="project-description">{{ repo.description or 'No description available' }}</p>
                            <div class="project-meta">
                                <span class="badge badge-stars">⭐ {{ repo.stars }}</span>
                                <span class="badge badge-forks">🔱 {{ repo.forks }}</span>
                                <span class="badge badge-score">Score: {{ repo.ai_ml_relevance_score|round(1) }}/10</span>
                                {% if repo.language %}
                                <span class="badge badge-language">{{ repo.language }}</span>
                                {% endif %}
                            </div>
                            {% if repo.topics %}
                            <div class="topics">
                                {% for topic in repo.topics[:5] %}
                                <span class="topic-tag">{{ topic }}</span>
                                {% endfor %}
                            </div>
                            {% endif %}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            {% endif %}
            
            <h2>💻 Languages</h2>
            <div class="languages">
                {% for lang, data in languages.items() %}
                <div class="language-item">
                    <div class="language-name">{{ lang }}</div>
                    <div class="language-stats">{{ data.count }} projects ({{ data.percentage|round(1) }}%)</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <footer>
            <p>Generated automatically by AI/ML Portfolio Scanner</p>
            <p>Powered by GitHub API</p>
        </footer>
    </div>
    
    <script>
        function toggleExpand() {
            const content = document.getElementById('allProjects');
            const button = document.getElementById('expandBtn');
            const isExpanded = content.classList.contains('expanded');
            
            if (isExpanded) {
                content.classList.remove('expanded');
                button.textContent = '📁 Show All {{ other_count }} More Projects';
            } else {
                content.classList.add('expanded');
                button.textContent = '📁 Hide Additional Projects';
            }
        }
        
        function scrollToProjects() {
            document.getElementById('projects').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>""")
        
        return template.render(
            date=portfolio.generation_date[:10],
            total_repos=portfolio.total_repositories,
            stars=portfolio.insights['total_stars'],
            forks=portfolio.insights['total_forks'],
            score=round(portfolio.expertise_metrics['overall_score'], 1),
            highlights=top_5,
            other_repos=other_repos,
            other_count=len(other_repos),
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
