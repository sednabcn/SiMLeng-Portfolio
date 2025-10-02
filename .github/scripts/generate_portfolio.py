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
from portfolio_builder import PortfolioBuilder
from data_models import RepositoryData, PortfolioData
from status_detector import determine_project_status
from config_loader import PortfolioConfig

def calculate_relevance_score(repo, frameworks):
    """Calculate AI/ML relevance score for a repository"""
    score = 0.0
    
    # Framework detection (strongest signal)
    score += len(frameworks.get('ml_frameworks', [])) * 2.0
    score += len(frameworks.get('llm_frameworks', [])) * 3.0
    
    # Keywords in repository name
    name_lower = repo['name'].lower()
    ml_keywords = [
        'ml', 'machine-learning', 'deep-learning', 'neural', 'ai',
        'tensorflow', 'pytorch', 'keras', 'scikit', 'sklearn',
        'data-science', 'datascience', 'nlp', 'computer-vision',
        'reinforcement-learning', 'time-series', 'statistics',
        'statsmodels', 'prediction', 'model', 'classifier',
        'regression', 'clustering', 'optimization', 'hadoop',
        'spark', 'analytics', 'visualization', 'notebook'
    ]
    
    for keyword in ml_keywords:
        if keyword in name_lower:
            score += 1.5
            break  # Only count once per repo
    
    # Keywords in description
    description = (repo.get('description') or '').lower()
    desc_keywords = [
        'machine learning', 'deep learning', 'neural network',
        'artificial intelligence', 'data science', 'nlp',
        'natural language', 'computer vision', 'time series',
        'prediction', 'classification', 'regression', 'model',
        'algorithm', 'training', 'dataset'
    ]
    
    for keyword in desc_keywords:
        if keyword in description:
            score += 1.0
            break  # Only count once per repo
    
    # Topics (GitHub tags)
    topics = repo.get('topics', [])
    ml_topics = {
        'machine-learning', 'deep-learning', 'artificial-intelligence',
        'data-science', 'nlp', 'computer-vision', 'neural-networks',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn'
    }
    
    topic_matches = len(set(topics) & ml_topics)
    score += topic_matches * 1.5
    
    # Language bonus (Python, R, Julia are common for ML)
    language = repo.get('language') or ''
    if language.lower() in ['python', 'jupyter notebook', 'r', 'julia']:
        score += 1.0
    
    # Popularity bonus
    stars = repo.get('stargazers_count', 0)
    if stars > 10:
        score += 2.0
    elif stars > 5:
        score += 1.0
    
    # Cap at 10.0
    return min(score, 10.0)

async def main():
    try:
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            print("ERROR: GITHUB_TOKEN not set")
            sys.exit(1)
        
        username = os.getenv('TARGET_USER', 'sednabcn')
        include_private = os.getenv('INCLUDE_PRIVATE_REPOS', 'false').lower() == 'true'
        
        print(f"Scanning repositories for user: {username}")
        print(f"Include private repos: {include_private}")
        
        # Load configuration
        config_path = Path(__file__).parent.parent / 'portfolio-config.yml'
        portfolio_config = PortfolioConfig(str(config_path))
        config = portfolio_config.config
        print(f"Loaded config from {config_path}")
        
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
        private_count = 0
        skipped_count = 0
        status_counts = {'current': 0, 'recent': 0, 'past': 0}
        
        for repo in repos:
            print(f"Analyzing: {repo['name']}")
            
            # Skip private repositories (unless enabled)
            if repo.get('private', False) and not include_private:
                print(f"  ✗ Skipped (private repository)")
                private_count += 1
                continue
            
            # Check blacklist/whitelist
            if portfolio_config.is_blacklisted(repo['name']):
                print(f"  ✗ Skipped (blacklisted)")
                skipped_count += 1
                continue
            if portfolio_config.get_whitelist() and not portfolio_config.is_whitelisted(repo['name']):
                print(f"  ✗ Skipped (not in whitelist)")
                skipped_count += 1
                continue
            
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
            
            # Calculate relevance score with improved logic
            score = calculate_relevance_score(repo, frameworks)
            
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
                pushed_at=repo.get('pushed_at', ''),  # ADD THIS LINE BACK
                frameworks=frameworks,
                code_analysis=code_analysis,
                ai_ml_relevance_score=score
            )
            
            # Determine project status
            repo_data.project_status = determine_project_status(repo_data, config)
            status_counts[repo_data.project_status] += 1
            
            # Lower threshold: include repos with score >= 1.0
            if repo_data.ai_ml_relevance_score >= 1.0:
                analyzed_repos.append(repo_data)
                print(f"  ✓ Included (score: {score:.1f}/10, status: {repo_data.project_status})")
            else:
                print(f"  ✗ Skipped (score: {score:.1f}/10)")
                skipped_count += 1
        
        print(f"\n{'='*60}")
        print(f"Repository Summary:")
        print(f"  Total scanned: {len(repos)}")
        print(f"  Private repos (excluded): {private_count}")
        print(f"  Low relevance (excluded): {skipped_count}")
        print(f"  Included in portfolio: {len(analyzed_repos)}")
        print(f"\nStatus Distribution:")
        print(f"  🟢 CURRENT: {status_counts['current']}")
        print(f"  🟡 RECENT: {status_counts['recent']}")
        print(f"  ⚪ PAST: {status_counts['past']}")
        print(f"{'='*60}\n")
        
        # Build portfolio
        print("Building portfolio...")
        config_dict = {
            'output': {'generate_html': True, 'include_code_samples': True}
        }
        portfolio = portfolio_builder.build_portfolio(analyzed_repos, config_dict)
        
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
        
        print(f"\n{'='*60}")
        print(f"SUCCESS! Portfolio generated at {output_dir}")
        print(f"Repositories analyzed: {len(analyzed_repos)}")
        print(f"Overall expertise score: {portfolio.expertise_metrics.get('overall_score', 0):.1f}/10")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
