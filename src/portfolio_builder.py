#!/usr/bin/env python3
"""
Portfolio Builder Module
Builds comprehensive AI/ML portfolio from analyzed repositories
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from jinja2 import Template
from dataclasses import asdict

class PortfolioBuilder:
    """Builds and formats AI/ML portfolio from repository analysis data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build_portfolio(self, repositories: List[Any], config: Dict[str, Any]) -> 'PortfolioData':
        """Build comprehensive portfolio from analyzed repositories."""
        self.logger.info(f"Building portfolio from {len(repositories)} repositories")
        
        portfolio = PortfolioData()
        
        # Basic statistics
        portfolio.total_repositories = len(repositories)
        portfolio.generation_date = datetime.now().isoformat()
        
        # Process repositories
        for repo in repositories:
            if repo.ai_ml_relevance_score >= 3.0:  # Only include relevant repos
                portfolio.repositories.append(repo)
        
        # Generate insights
        portfolio.insights = self._generate_insights(portfolio.repositories)
        
        # Categorize repositories
        portfolio.categories = self._categorize_repositories(portfolio.repositories)
        
        # Extract skills and technologies
        portfolio.skills = self._extract_skills(portfolio.repositories)
        
        # Generate project highlights
        portfolio.highlights = self._select_highlights(portfolio.repositories, max_highlights=5)
        
        # Calculate expertise metrics
        portfolio.expertise_metrics = self._calculate_expertise_metrics(portfolio.repositories)
        
        return portfolio
    
    def _generate_insights(self, repositories: List[Any]) -> Dict[str, Any]:
        """Generate insights from repository analysis."""
        insights = {
            'total_stars': sum(repo.stars for repo in repositories),
            'total_forks': sum(repo.forks for repo in repositories),
            'languages': {},
            'frameworks_summary': {},
            'activity_timeline': [],
            'collaboration_score': 0.0,
            'code_quality_score': 0.0
        }
        
        # Language distribution
        language_counts = {}
        for repo in repositories:
            if repo.language:
                language_counts[repo.language] = language_counts.get(repo.language, 0) + 1
        
        total_repos = len(repositories)
        insights['languages'] = {
            lang: {"count": count, "percentage": (count / total_repos) * 100}
            for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True)
        }
        
        # Framework summary
        framework_counts = {}
        for repo in repositories:
            for category, frameworks in repo.frameworks.items():
                for framework in frameworks:
                    name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                    framework_counts[name] = framework_counts.get(name, 0) + 1
        
        insights['frameworks_summary'] = dict(
            sorted(framework_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        )
        
        # Activity timeline
        activity_by_year = {}
        for repo in repositories:
            if repo.created_at:
                year = repo.created_at[:4]  # Extract year
                activity_by_year[year] = activity_by_year.get(year, 0) + 1
        
        insights['activity_timeline'] = [
            {"year": year, "repositories": count}
            for year, count in sorted(activity_by_year.items())
        ]
        
        # Collaboration score (based on forks and contributions)
        if repositories:
            avg_forks = insights['total_forks'] / len(repositories)
            insights['collaboration_score'] = min(avg_forks / 10.0 * 10, 10.0)
        
        return insights
    
    def _categorize_repositories(self, repositories: List[Any]) -> Dict[str, List[Dict]]:
        """Categorize repositories by type and domain."""
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
            repo_dict = repo.to_dict() if hasattr(repo, 'to_dict') else asdict(repo)
            
            # Categorize based on frameworks and content
            frameworks = repo.frameworks
            description = repo.description.lower() if repo.description else ""
            topics = [topic.lower() for topic in repo.topics] if repo.topics else []
            
            # Machine Learning
            if any(fw in frameworks.get('ml_frameworks', []) for fw in ['scikit-learn', 'xgboost']):
                categories['machine_learning'].append(repo_dict)
            
            # Deep Learning
            if any(fw in frameworks.get('ml_frameworks', []) + frameworks.get('deep_learning', []) 
                   for fw in ['tensorflow', 'pytorch', 'keras']):
                categories['deep_learning'].append(repo_dict)
            
            # NLP
            nlp_indicators = ['nlp', 'natural language', 'text', 'language model', 'bert', 'gpt']
            if (any(ind in description for ind in nlp_indicators) or 
                any(ind in topics for ind in nlp_indicators) or
                any(fw in frameworks.get('llm_frameworks', []) for fw in ['transformers', 'openai'])):
                categories['natural_language_processing'].append(repo_dict)
            
            # Computer Vision  
            cv_indicators = ['vision', 'image', 'opencv', 'computer vision', 'cnn', 'detection']
            if (any(ind in description for ind in cv_indicators) or 
                any(ind in topics for ind in cv_indicators) or
                any(fw in frameworks.get('computer_vision', []) for fw in ['opencv'])):
                categories['computer_vision'].append(repo_dict)
            
            # Data Science
            if (any(fw in frameworks.get('data_frameworks', []) for fw in ['pandas', 'numpy']) or
                'data science' in description or 'data-science' in topics):
                categories['data_science'].append(repo_dict)
            
            # Research (based on academic indicators)
            research_indicators = ['paper', 'research', 'experiment', 'study', 'analysis']
            if (any(ind in description for ind in research_indicators) or
                any(ind in topics for ind in research_indicators) or
                repo.code_analysis.get('has_notebooks', False)):
                categories['research'].append(repo_dict)
            
            # Production (based on deployment indicators)
            prod_indicators = ['api', 'deploy', 'production', 'docker', 'kubernetes']
            if any(ind in description for ind in prod_indicators):
                categories['production'].append(repo_dict)
            
            # Educational
            edu_indicators = ['tutorial', 'example', 'demo', 'course', 'learning']
            if any(ind in description for ind in edu_indicators):
                categories['educational'].append(repo_dict)
        
        # Remove empty categories and sort by repository count
        return {k: v for k, v in categories.items() if v}
    
    def _extract_skills(self, repositories: List[Any]) -> Dict[str, Any]:
        """Extract skills and technologies from repositories."""
        skills = {
            'programming_languages': {},
            'ml_frameworks': {},
            'tools_and_libraries': {},
            'domains': {},
            'techniques': set(),
            'proficiency_levels': {}
        }
        
        for repo in repositories:
            # Programming languages
            if repo.language:
                skills['programming_languages'][repo.language] = (
                    skills['programming_languages'].get(repo.language, 0) + 1
                )
            
            # ML Frameworks
            for category, frameworks in repo.frameworks.items():
                for framework in frameworks:
                    name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                    skills['ml_frameworks'][name] = skills['ml_frameworks'].get(name, 0) + 1
            
            # Extract techniques from code analysis
            code_patterns = repo.code_analysis.get('code_patterns', {})
            for pattern, count in code_patterns.items():
                if count > 0:
                    skills['techniques'].add(pattern.replace('_', ' ').title())
        
        # Calculate proficiency levels
        total_repos = len(repositories)
        for framework, count in skills['ml_frameworks'].items():
            usage_ratio = count / total_repos
            if usage_ratio >= 0.3:
                skills['proficiency_levels'][framework] = 'Advanced'
            elif usage_ratio >= 0.15:
                skills['proficiency_levels'][framework] = 'Intermediate'
            else:
                skills['proficiency_levels'][framework] = 'Beginner'
        
        # Convert set to list for JSON serialization
        skills['techniques'] = list(skills['techniques'])
        
        return skills
    
    def _select_highlights(self, repositories: List[Any], max_highlights: int = 5) -> List[Dict[str, Any]]:
        """Select top repositories as portfolio highlights."""
        # Sort repositories by relevance score and stars
        sorted_repos = sorted(
            repositories,
            key=lambda x: (x.ai_ml_relevance_score, x.stars),
            reverse=True
        )
        
        highlights = []
        for repo in sorted_repos[:max_highlights]:
            highlight = {
                'name': repo.name,
                'full_name': repo.full_name,
                'description': repo.description,
                'url': repo.url,
                'stars': repo.stars,
                'forks': repo.forks,
                'language': repo.language,
                'topics': repo.topics,
                'relevance_score': repo.ai_ml_relevance_score,
                'key_frameworks': self._get_key_frameworks(repo.frameworks),
                'highlights': self._generate_repo_highlights(repo)
            }
            highlights.append(highlight)
        
        return highlights
    
    def _get_key_frameworks(self, frameworks: Dict[str, List]) -> List[str]:
        """Get the most important frameworks for a repository."""
        key_frameworks = []
        
        # Priority order for framework categories
        priority_categories = ['llm_frameworks', 'ml_frameworks', 'deep_learning', 'computer_vision']
        
        for category in priority_categories:
            category_frameworks = frameworks.get(category, [])
            for framework in category_frameworks[:2]:  # Top 2 from each category
                name = framework if isinstance(framework, str) else framework.get('name', 'Unknown')
                if name not in key_frameworks:
                    key_frameworks.append(name)
        
        return key_frameworks[:5]  # Limit to top 5
    
    def _generate_repo_highlights(self, repo: Any) -> List[str]:
        """Generate key highlights for a repository."""
        highlights = []
        
        # Stars and forks
        if repo.stars > 100:
            highlights.append(f"{repo.stars} GitHub stars")
        if repo.forks > 20:
            highlights.append(f"{repo.forks} forks")
        
        # Code analysis highlights
        code_analysis = repo.code_analysis
        if code_analysis.get('has_notebooks'):
            highlights.append("Jupyter notebook examples")
        if code_analysis.get('has_models'):
            highlights.append("Pre-trained models included")
        if code_analysis.get('has_tests'):
            highlights.append("Comprehensive test suite")
        
        # Framework highlights
        ml_frameworks = repo.frameworks.get('ml_frameworks', [])
        if len(ml_frameworks) >= 3:
            highlights.append("Multi-framework implementation")
        
        # AI/ML score
        if repo.ai_ml_relevance_score >= 8.0:
            highlights.append("High AI/ML relevance")
        
        return highlights[:4]  # Limit to top 4 highlights
    
    def _calculate_expertise_metrics(self, repositories: List[Any]) -> Dict[str, float]:
        """Calculate expertise metrics across different areas."""
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
        ml_repos = [r for r in repositories if any(
            fw in r.frameworks.get('ml_frameworks', []) 
            for fw in ['scikit-learn', 'xgboost', 'lightgbm']
        )]
        metrics['machine_learning'] = len(ml_repos) / total_repos * 10
        
        # Deep Learning expertise
        dl_repos = [r for r in repositories if any(
            fw in r.frameworks.get('ml_frameworks', []) + r.frameworks.get('deep_learning', [])
            for fw in ['tensorflow', 'pytorch', 'keras']
        )]
        metrics['deep_learning'] = len(dl_repos) / total_repos * 10
        
        # Data Science expertise
        ds_repos = [r for r in repositories if any(
            fw in r.frameworks.get('data_frameworks', [])
            for fw in ['pandas', 'numpy']
        )]
        metrics['data_science'] = len(ds_repos) / total_repos * 10
        
        # Software Engineering (based on code quality indicators)
        se_score = 0
        for repo in repositories:
            if repo.code_analysis.get('has_tests'):
                se_score += 1
            if repo.code_analysis.get('quality_metrics', {}).get('documentation', 0) > 0:
                se_score += 0.5
        metrics['software_engineering'] = (se_score / total_repos) * 10
        
        # Research ability (based on notebooks, documentation, complexity)
        research_score = sum(1 for repo in repositories if repo.code_analysis.get('has_notebooks'))
        metrics['research_ability'] = (research_score / total_repos) * 10
        
        # Collaboration (based on forks and community engagement)
        total_forks = sum(repo.forks for repo in repositories)
        avg_forks = total_forks / total_repos if total_repos > 0 else 0
        metrics['collaboration'] = min(avg_forks / 5.0 * 10, 10.0)
        
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
        
        # Cap all scores at 10.0
        for key in metrics:
            metrics[key] = min(metrics[key], 10.0)
        
        return metrics
    
    def generate_html_report(self, portfolio: 'PortfolioData') -> str:
        """Generate HTML portfolio report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI/ML Portfolio Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        .section {
            margin: 40px 0;
            padding: 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }
        .highlight-card {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .highlight-title {
            color: #2980b9;
            text-decoration: none;
            font-weight: bold;
            font-size: 1.2em;
        }
        .highlight-meta {
            color: #7f8c8d;
            font-size: 0.9em;
            margin: 10px 0;
        }
        .tags {
            margin: 10px 0;
        }
        .tag {
            background: #e74c3c;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-right: 5px;
        }
        .framework-tag {
            background: #27ae60;
        }
        .skill-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .skill-category {
            background: white;
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #ddd;
        }
        .skill-bar {
            background: #ecf0f1;
            border-radius: 10px;
            height: 20px;
            margin: 5px 0;
            overflow: hidden;
        }
        .skill-progress {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        .expertise-radar {
            text-align: center;
            margin: 30px 0;
        }
        .chart-placeholder {
            background: #f8f9fa;
            border: 2px dashed #dee2e6;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6c757d;
            border-radius: 8px;
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
                <div class="stat-label">Total Repositories</div>
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
                <div class="stat-value">{{ portfolio.expertise_metrics.overall_score|round(1) }}</div>
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
                <p>{{ highlight.description }}</p>
                <div class="tags">
                    {% for framework in highlight.key_frameworks %}
                    <span class="tag framework-tag">{{ framework }}</span>
                    {% endfor %}
                    {% for topic in highlight.topics[:3] %}
                    <span class="tag">{{ topic }}</span>
                    {% endfor %}
                </div>
                <div>
                    {% for point in highlight.highlights %}
                    <li>{{ point }}</li>
                    {% endfor %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>🛠️ Technical Skills</h2>
            <div class="skill-grid">
                <div class="skill-category">
                    <h3>Programming Languages</h3>
                    {% for lang, data in portfolio.skills.programming_languages.items() %}
                    <div>
                        <span>{{ lang }}</span>
                        <div class="skill-bar">
                            <div class="skill-progress" style="width: {{ data.percentage }}%"></div>
                        </div>
                        <small>{{ data.count }} repositories ({{ data.percentage|round(1) }}%)</small>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="skill-category">
                    <h3>ML Frameworks</h3>
                    {% for framework, count in portfolio.skills.ml_frameworks.items() %}
                    <div>
                        <span>{{ framework }}</span>
                        <div class="skill-bar">
                            <div class="skill-progress" style="width: {{ (count / portfolio.total_repositories * 100) }}%"></div>
                        </div>
                        <small>{{ count }} projects • {{ portfolio.skills.proficiency_levels.get(framework, 'Beginner') }}</small>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Expertise Metrics</h2>
            <div class="expertise-radar">
                <div class="chart-placeholder">
                    Expertise Radar Chart<br>
                    ML: {{ portfolio.expertise_metrics.machine_learning|round(1) }}/10 |
                    DL: {{ portfolio.expertise_metrics.deep_learning|round(1) }}/10 |
                    DS: {{ portfolio.expertise_metrics.data_science|round(1) }}/10<br>
                    SE: {{ portfolio.expertise_metrics.software_engineering|round(1) }}/10 |
                    Research: {{ portfolio.expertise_metrics.research_ability|round(1) }}/10 |
                    Collaboration: {{ portfolio.expertise_metrics.collaboration|round(1) }}/10
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Repository Categories</h2>
            {% for category, repos in portfolio.categories.items() %}
            <h3>{{ category.replace('_', ' ').title() }} ({{ repos|length }} repositories)</h3>
            <div style="margin-left: 20px;">
                {% for repo in repos[:3] %}
                <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                    <strong><a href="{{ repo.url }}" target="_blank">{{ repo.name }}</a></strong>
                    <span style="color: #7f8c8d;"> - {{ repo.description[:100] }}{% if repo.description|length > 100 %}...{% endif %}</span>
                </div>
                {% endfor %}
                {% if repos|length > 3 %}
                <p><em>... and {{ repos|length - 3 }} more repositories</em></p>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>📅 Activity Timeline</h2>
            {% for activity in portfolio.insights.activity_timeline %}
            <div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee;">
                <span>{{ activity.year }}</span>
                <span>{{ activity.repositories }} repositories</span>
            </div>
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>🔍 Framework Usage</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">
                {% for framework, count in portfolio.insights.frameworks_summary.items() %}
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">
                    <strong>{{ framework }}</strong><br>
                    <span style="color: #3498db; font-size: 1.2em;">{{ count }}</span> projects
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        return template.render(portfolio=portfolio)
    
    def generate_markdown_summary(self, portfolio: 'PortfolioData') -> str:
        """Generate markdown summary of the portfolio."""
        md_content = f"""# 🤖 AI/ML Portfolio Summary

*Generated on {portfolio.generation_date[:10]}*

## 📊 Overview

- **Total Repositories:** {portfolio.total_repositories}
- **GitHub Stars:** {portfolio.insights['total_stars']}
- **Total Forks:** {portfolio.insights['total_forks']}
- **Overall Expertise Score:** {portfolio.expertise_metrics['overall_score']:.1f}/10

## 🌟 Portfolio Highlights

"""
        
        for highlight in portfolio.highlights[:3]:
            md_content += f"""### [{highlight['name']}]({highlight['url']})
⭐ {highlight['stars']} stars • 🍴 {highlight['forks']} forks • 📊 {highlight['relevance_score']:.1f}/10 relevance

{highlight['description']}

**Key Frameworks:** {', '.join(highlight['key_frameworks'])}

**Highlights:**
{chr(10).join(f"- {point}" for point in highlight['highlights'])}

---

"""
        
        md_content += """## 🛠️ Technical Skills

### Programming Languages
"""
        
        for lang, data in list(portfolio.skills['programming_languages'].items())[:5]:
            md_content += f"- **{lang}**: {data['count']} repositories ({data['percentage']:.1f}%)\n"
        
        md_content += "\n### ML/AI Frameworks\n"
        
        for framework, count in list(portfolio.skills['ml_frameworks'].items())[:10]:
            proficiency = portfolio.skills['proficiency_levels'].get(framework, 'Beginner')
            md_content += f"- **{framework}**: {count} projects ({proficiency})\n"
        
        md_content += f"""
## 📈 Expertise Breakdown

- **Machine Learning:** {portfolio.expertise_metrics['machine_learning']:.1f}/10
- **Deep Learning:** {portfolio.expertise_metrics['deep_learning']:.1f}/10  
- **Data Science:** {portfolio.expertise_metrics['data_science']:.1f}/10
- **Software Engineering:** {portfolio.expertise_metrics['software_engineering']:.1f}/10
- **Research Ability:** {portfolio.expertise_metrics['research_ability']:.1f}/10
- **Collaboration:** {portfolio.expertise_metrics['collaboration']:.1f}/10

## 📂 Repository Categories

"""
        
        for category, repos in portfolio.categories.items():
            category_name = category.replace('_', ' ').title()
            md_content += f"### {category_name} ({len(repos)} repositories)\n\n"
            
            for repo in repos[:3]:
                md_content += f"- **[{repo['name']}]({repo['url']})** - {repo['description'][:100]}{'...' if len(repo['description']) > 100 else ''}\n"
            
            if len(repos) > 3:
                md_content += f"- *... and {len(repos) - 3} more repositories*\n"
            
            md_content += "\n"
        
        md_content += """## 🚀 Key Achievements

"""
        
        # Add some calculated achievements
        total_stars = portfolio.insights['total_stars']
        if total_stars > 500:
            md_content += f"- 🌟 Accumulated over {total_stars} GitHub stars across projects\n"
        
        if portfolio.insights['total_forks'] > 100:
            md_content += f"- 🍴 Projects forked {portfolio.insights['total_forks']} times by the community\n"
        
        # Framework diversity
        unique_frameworks = len(portfolio.skills['ml_frameworks'])
        if unique_frameworks >= 5:
            md_content += f"- 🛠️ Experience with {unique_frameworks} different ML/AI frameworks\n"
        
        # Multi-category expertise
        categories_with_projects = len([cat for cat, repos in portfolio.categories.items() if len(repos) >= 2])
        if categories_with_projects >= 3:
            md_content += f"- 🎯 Expertise across {categories_with_projects} different AI/ML domains\n"
        
        return md_content
    
    def export_json(self, portfolio: 'PortfolioData', filepath: str):
        """Export portfolio data as JSON."""
        with open(filepath, 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Portfolio exported to {filepath}")
    
    def generate_skills_matrix(self, portfolio: 'PortfolioData') -> Dict[str, Any]:
        """Generate a detailed skills matrix."""
        matrix = {
            'technical_skills': {},
            'domain_expertise': {},
            'proficiency_levels': {},
            'growth_areas': []
        }
        
        # Technical skills matrix
        frameworks = portfolio.skills['ml_frameworks']
        total_repos = portfolio.total_repositories
        
        for framework, count in frameworks.items():
            usage_ratio = count / total_repos if total_repos > 0 else 0
            
            matrix['technical_skills'][framework] = {
                'usage_count': count,
                'usage_percentage': usage_ratio * 100,
                'proficiency_level': portfolio.skills['proficiency_levels'].get(framework, 'Beginner'),
                'experience_level': self._calculate_experience_level(usage_ratio)
            }
        
        # Domain expertise
        for category, repos in portfolio.categories.items():
            if repos:
                matrix['domain_expertise'][category] = {
                    'project_count': len(repos),
                    'expertise_score': (len(repos) / total_repos) * 10 if total_repos > 0 else 0,
                    'key_projects': [repo['name'] for repo in repos[:3]]
                }
        
        # Identify growth areas
        all_categories = ['machine_learning', 'deep_learning', 'natural_language_processing', 
                         'computer_vision', 'data_science']
        
        for category in all_categories:
            if category not in portfolio.categories or len(portfolio.categories[category]) < 2:
                matrix['growth_areas'].append(category)
        
        return matrix
    
    def _calculate_experience_level(self, usage_ratio: float) -> str:
        """Calculate experience level based on usage ratio."""
        if usage_ratio >= 0.4:
            return 'Expert'
        elif usage_ratio >= 0.2:
            return 'Proficient'  
        elif usage_ratio >= 0.1:
            return 'Competent'
        else:
            return 'Novice'


class PortfolioData:
    """Data class for portfolio information."""
    
    def __init__(self):
        self.total_repositories: int = 0
        self.generation_date: str = ""
        self.repositories: List[Any] = []
        self.insights: Dict[str, Any] = {}
        self.categories: Dict[str, List[Dict]] = {}
        self.skills: Dict[str, Any] = {}
        self.highlights: List[Dict[str, Any]] = []
        self.expertise_metrics: Dict[str, float] = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_repositories': self.total_repositories,
            'generation_date': self.generation_date,
            'repositories': [repo.to_dict() if hasattr(repo, 'to_dict') else asdict(repo) 
                           for repo in self.repositories],
            'insights': self.insights,
            'categories': self.categories,
            'skills': self.skills,
            'highlights': self.highlights,
            'expertise_metrics': self.expertise_metrics
        }
