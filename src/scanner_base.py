#!/usr/bin/env python3
"""
GitHub AI/ML/LLM Portfolio Scanner
Main orchestration script for scanning repositories and building portfolio data

Usage:
    python github_scanner.py --user USERNAME --token GITHUB_TOKEN
    python github_scanner.py --repos repo1,repo2,repo3 --token GITHUB_TOKEN
    python github_scanner.py --config config.yaml
"""

import os
import sys
import json
import yaml
import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set

# Import our custom modules
from repo_analyzer import RepoAnalyzer
from code_analyzer import CodeAnalyzer
from framework_detector import FrameworkDetector
from portfolio_builder import PortfolioBuilder
from data_models import RepositoryData, PortfolioData

class GitHubPortfolioScanner:
    """Main class for orchestrating the GitHub scanning and portfolio building process."""
    
    def __init__(self, github_token: str, config_path: Optional[str] = None):
        self.github_token = github_token
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components
        self.repo_analyzer = RepoAnalyzer(github_token)
        self.code_analyzer = CodeAnalyzer()
        self.framework_detector = FrameworkDetector()
        self.portfolio_builder = PortfolioBuilder()
        
        # Results storage
        self.scanned_repos: List[RepositoryData] = []
        self.portfolio_data: PortfolioData = PortfolioData()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'frameworks': {
                'ml_frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'jax'],
                'llm_frameworks': ['transformers', 'openai', 'langchain', 'llamaindex', 'anthropic'],
                'dl_frameworks': ['torch', 'tensorflow', 'mxnet', 'caffe', 'theano'],
                'data_frameworks': ['pandas', 'numpy', 'dask', 'polars', 'spark'],
                'viz_frameworks': ['matplotlib', 'plotly', 'seaborn', 'bokeh', 'altair']
            },
            'file_patterns': {
                'notebooks': ['*.ipynb'],
                'python': ['*.py'],
                'configs': ['*.yaml', '*.yml', '*.json', 'requirements.txt', 'environment.yml'],
                'docs': ['*.md', '*.rst', '*.txt'],
                'models': ['*.pkl', '*.joblib', '*.h5', '*.pt', '*.pth', '*.onnx']
            },
            'analysis': {
                'max_file_size': 1048576,  # 1MB
                'max_repos': 100,
                'include_forks': False,
                'min_stars': 0,
                'exclude_archived': True
            },
            'output': {
                'format': 'json',
                'include_code_samples': True,
                'max_code_sample_lines': 50,
                'generate_html': True,
                'generate_pdf': False
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                return {**default_config, **user_config}
        
        return default_config
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'github_scanner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def scan_user_repositories(self, username: str) -> List[RepositoryData]:
        """Scan all repositories for a given user."""
        self.logger.info(f"Scanning repositories for user: {username}")
        
        try:
            repos = await self.repo_analyzer.get_user_repositories(
                username, 
                max_repos=self.config['analysis']['max_repos'],
                include_forks=self.config['analysis']['include_forks']
            )
            
            filtered_repos = []
            for repo in repos:
                if self._should_analyze_repo(repo):
                    filtered_repos.append(repo)
            
            self.logger.info(f"Found {len(filtered_repos)} repositories to analyze")
            
            # Analyze repositories concurrently
            analyzed_repos = await asyncio.gather(*[
                self._analyze_repository(repo) for repo in filtered_repos
            ])
            
            return [repo for repo in analyzed_repos if repo is not None]
            
        except Exception as e:
            self.logger.error(f"Error scanning user repositories: {e}")
            return []
    
    async def scan_specific_repositories(self, repo_urls: List[str]) -> List[RepositoryData]:
        """Scan specific repositories by URL."""
        self.logger.info(f"Scanning {len(repo_urls)} specific repositories")
        
        analyzed_repos = await asyncio.gather(*[
            self._analyze_repository_by_url(url) for url in repo_urls
        ])
        
        return [repo for repo in analyzed_repos if repo is not None]
    
    def _should_analyze_repo(self, repo_data: Dict) -> bool:
        """Determine if a repository should be analyzed based on config criteria."""
        if repo_data.get('archived') and self.config['analysis']['exclude_archived']:
            return False
        
        if repo_data.get('stargazers_count', 0) < self.config['analysis']['min_stars']:
            return False
        
        # Check for AI/ML indicators in description or topics
        description = repo_data.get('description', '').lower()
        topics = [topic.lower() for topic in repo_data.get('topics', [])]
        
        ai_ml_keywords = {
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'llm', 'nlp', 'computer vision', 'data science', 'tensorflow', 'pytorch',
            'transformer', 'gpt', 'bert', 'lstm', 'cnn', 'rnn', 'reinforcement learning'
        }
        
        text_to_check = description + ' ' + ' '.join(topics)
        return any(keyword in text_to_check for keyword in ai_ml_keywords)
    
    async def _analyze_repository(self, repo_data: Dict) -> Optional[RepositoryData]:
        """Analyze a single repository."""
        repo_name = repo_data['full_name']
        self.logger.info(f"Analyzing repository: {repo_name}")
        
        try:
            # Get repository contents
            contents = await self.repo_analyzer.get_repository_contents(repo_name)
            
            # Detect frameworks
            frameworks = await self.framework_detector.detect_frameworks(contents)
            
            # Analyze code
            code_analysis = await self.code_analyzer.analyze_repository_code(contents)
            
            # Create repository data object
            repo_obj = RepositoryData(
                name=repo_data['name'],
                full_name=repo_data['full_name'],
                description=repo_data.get('description', ''),
                url=repo_data['html_url'],
                stars=repo_data.get('stargazers_count', 0),
                forks=repo_data.get('forks_count', 0),
                language=repo_data.get('language'),
                topics=repo_data.get('topics', []),
                created_at=repo_data.get('created_at'),
                updated_at=repo_data.get('updated_at'),
                frameworks=frameworks,
                code_analysis=code_analysis,
                ai_ml_relevance_score=self._calculate_relevance_score(
                    repo_data, frameworks, code_analysis
                )
            )
            
            return repo_obj
            
        except Exception as e:
            self.logger.error(f"Error analyzing repository {repo_name}: {e}")
            return None
    
    async def _analyze_repository_by_url(self, repo_url: str) -> Optional[RepositoryData]:
        """Analyze a repository by its URL."""
        # Extract owner/repo from URL
        parts = repo_url.strip('/').split('/')
        if len(parts) < 2:
            self.logger.error(f"Invalid repository URL: {repo_url}")
            return None
        
        repo_name = f"{parts[-2]}/{parts[-1]}"
        
        # Get repository metadata
        repo_data = await self.repo_analyzer.get_repository_metadata(repo_name)
        if not repo_data:
            return None
        
        return await self._analyze_repository(repo_data)
    
    def _calculate_relevance_score(self, repo_data: Dict, frameworks: Dict, 
                                 code_analysis: Dict) -> float:
        """Calculate AI/ML relevance score for a repository."""
        score = 0.0
        
        # Framework detection score
        ml_frameworks = frameworks.get('ml_frameworks', [])
        llm_frameworks = frameworks.get('llm_frameworks', [])
        dl_frameworks = frameworks.get('dl_frameworks', [])
        
        score += len(ml_frameworks) * 2.0
        score += len(llm_frameworks) * 3.0
        score += len(dl_frameworks) * 2.5
        
        # Code analysis score
        if code_analysis.get('has_notebooks'):
            score += 1.5
        
        if code_analysis.get('has_models'):
            score += 2.0
        
        # Repository metadata score
        topics = repo_data.get('topics', [])
        ai_topics = ['machine-learning', 'deep-learning', 'artificial-intelligence', 
                    'neural-network', 'nlp', 'computer-vision', 'data-science']
        score += sum(1.0 for topic in topics if topic in ai_topics)
        
        # Description keywords
        description = repo_data.get('description', '').lower()
        ai_keywords = ['ml', 'ai', 'neural', 'model', 'training', 'prediction']
        score += sum(0.5 for keyword in ai_keywords if keyword in description)
        
        # Popularity bonus
        stars = repo_data.get('stargazers_count', 0)
        if stars > 100:
            score += min(stars / 100, 5.0)
        
        return min(score, 10.0)  # Cap at 10.0
    
    def build_portfolio(self) -> PortfolioData:
        """Build portfolio data from scanned repositories."""
        self.logger.info("Building portfolio from scanned repositories")
        
        # Sort repositories by relevance score
        sorted_repos = sorted(
            self.scanned_repos, 
            key=lambda x: x.ai_ml_relevance_score, 
            reverse=True
        )
        
        # Build portfolio
        portfolio = self.portfolio_builder.build_portfolio(
            sorted_repos, 
            self.config
        )
        
        return portfolio
    
    def save_results(self, output_dir: str = "portfolio_output"):
        """Save analysis results and portfolio."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw repository data
        with open(output_path / f"repositories_{timestamp}.json", 'w') as f:
            json.dump([repo.to_dict() for repo in self.scanned_repos], f, indent=2)
        
        # Save portfolio data
        portfolio = self.build_portfolio()
        with open(output_path / f"portfolio_{timestamp}.json", 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2)
        
        # Generate HTML report if configured
        if self.config['output']['generate_html']:
            html_report = self.portfolio_builder.generate_html_report(portfolio)
            with open(output_path / f"portfolio_report_{timestamp}.html", 'w') as f:
                f.write(html_report)
        
        # Generate markdown summary
        md_summary = self.portfolio_builder.generate_markdown_summary(portfolio)
        with open(output_path / f"portfolio_summary_{timestamp}.md", 'w') as f:
            f.write(md_summary)
        
        self.logger.info(f"Results saved to {output_path}")
        return output_path

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GitHub AI/ML Portfolio Scanner')
    parser.add_argument('--user', help='GitHub username to scan')
    parser.add_argument('--repos', help='Comma-separated list of repository URLs')
    parser.add_argument('--token', required=True, help='GitHub API token')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', default='portfolio_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = GitHubPortfolioScanner(args.token, args.config)
    
    try:
        if args.user:
            # Scan user repositories
            repos = await scanner.scan_user_repositories(args.user)
            scanner.scanned_repos.extend(repos)
        
        if args.repos:
            # Scan specific repositories
            repo_urls = [url.strip() for url in args.repos.split(',')]
            repos = await scanner.scan_specific_repositories(repo_urls)
            scanner.scanned_repos.extend(repos)
        
        if not args.user and not args.repos:
            print("Please specify either --user or --repos")
            return
        
        # Save results
        output_path = scanner.save_results(args.output)
        print(f"Portfolio scan completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"Found {len(scanner.scanned_repos)} relevant repositories")
        
    except Exception as e:
        scanner.logger.error(f"Error during scan: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
