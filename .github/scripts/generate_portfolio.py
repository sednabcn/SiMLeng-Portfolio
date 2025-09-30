#!/usr/bin/env python3
"""
Generate portfolio from GitHub repositories
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from repo_analyzer import RepoAnalyzer
from code_analyzer import CodeAnalyzer
from framework_detector import FrameworkDetector
from portfolio_builder import PortfolioBuilder, PortfolioData
from data_models import RepositoryData

async def main():
    try:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("ERROR: GITHUB_TOKEN not set")
            sys.exit(1)
        
        username = os.getenv('TARGET_USER', 'sednabcn')
        print(f"Scanning repositories for user: {username}")
        
        # Initialize components
        repo_analyzer = RepoAnalyzer(token)
        code_analyzer = CodeAnalyzer()
        framework_detector = FrameworkDetector()
        portfolio_builder = PortfolioBuilder()
        
        # Get repositories
        print("Fetching repositories...")
        repos = await repo_analyzer.get_user_repositories(username, max_repos=50)
        print(f"Found {len(repos)} repositories")
        
        # Analyze each repository
        analyzed_repos = []
        for repo in repos:
            print(f"Analyzing: {repo['name']}")
            
            # Detect frameworks
            frameworks = await framework_detector.detect_frameworks(repo)
            
            # Create minimal code analysis
            code_analysis = {
                'file_statistics': {'python_files': 0},
                'code_patterns': {},
                'has_notebooks': False,
                'has_models': False,
                'has_tests': False
            }
            
            # Calculate relevance score
            score = 0.0
            score += len(frameworks.get('ml_frameworks', [])) * 2.0
            score += len(frameworks.get('llm_frameworks', [])) * 3.0
            if repo.get('stargazers_count', 0) > 10:
                score += 2.0
            
            # Create repository data
            repo_data = RepositoryData(
                name=repo['name'],
                full_name=repo['full_name'],
                description=repo.get('description', ''),
                url=repo['html_url'],
                stars=repo.get('stargazers_count', 0),
                forks=repo.get('forks_count', 0),
                language=repo.get('language'),
                topics=repo.get('topics', []),
                created_at=repo.get('created_at', ''),
                updated_at=repo.get('updated_at', ''),
                frameworks=frameworks,
                code_analysis=code_analysis,
                ai_ml_relevance_score=min(score, 10.0)
            )
            
            if repo_data.ai_ml_relevance_score >= 3.0:
                analyzed_repos.append(repo_data)
        
        print(f"Found {len(analyzed_repos)} relevant AI/ML repositories")
        
        # Build portfolio
        print("Building portfolio...")
        config = {
            'output': {'generate_html': True, 'include_code_samples': True}
        }
        portfolio = portfolio_builder.build_portfolio(analyzed_repos, config)
        
        # Save results
        output_dir = Path(__file__).parent.parent.parent / 'docs'
        output_dir.mkdir(exist_ok=True)
        
        print("Generating HTML report...")
        html = portfolio_builder.generate_html_report(portfolio)
        (output_dir / 'index.html').write_text(html, encoding='utf-8')
        
        print("Generating markdown summary...")
        md = portfolio_builder.generate_markdown_summary(portfolio)
        (output_dir / 'README.md').write_text(md, encoding='utf-8')
        
        print("Saving portfolio data...")
        with open(output_dir / 'portfolio.json', 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2, default=str)
        
        print(f"SUCCESS! Portfolio generated at {output_dir}")
        print(f"Repositories analyzed: {len(analyzed_repos)}")
        print(f"Overall expertise score: {portfolio.expertise_metrics.get('overall_score', 0):.1f}/10")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
