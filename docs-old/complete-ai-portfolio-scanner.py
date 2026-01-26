# Complete AI Portfolio Scanner Workflow
# Missing modules and orchestration

# =================== MISSING MODULES ===================

# repo_analyzer.py
"""
GitHub Repository Analyzer - Handles GitHub API interactions
"""
import aiohttp
import asyncio
import base64
import json
from typing import Dict, List, Optional, Any
import logging
from urllib.parse import urlparse

class RepoAnalyzer:
    """GitHub API client for repository analysis."""
    
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Portfolio-Scanner/1.0"
        }
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.rate_limit_remaining = 5000
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_user_repositories(self, username: str, max_repos: int = 100, 
                                  include_forks: bool = False) -> List[Dict]:
        """Get all repositories for a user."""
        repos = []
        page = 1
        per_page = min(100, max_repos)
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            while len(repos) < max_repos:
                url = f"{self.base_url}/users/{username}/repos"
                params = {
                    "page": page,
                    "per_page": per_page,
                    "sort": "updated",
                    "direction": "desc"
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        batch = await response.json()
                        if not batch:
                            break
                            
                        for repo in batch:
                            if include_forks or not repo.get('fork', False):
                                repos.append(repo)
                                if len(repos) >= max_repos:
                                    break
                        
                        page += 1
                        await asyncio.sleep(0.1)  # Rate limiting
                    else:
                        self.logger.error(f"Failed to fetch repos: {response.status}")
                        break
        
        return repos[:max_repos]
    
    async def get_repository_contents(self, repo_full_name: str) -> Dict[str, Any]:
        """Get repository contents and file structure."""
        contents = {
            "files": [],
            "directories": [],
            "readme": None,
            "requirements": None,
            "notebooks": []
        }
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Get root contents
            url = f"{self.base_url}/repos/{repo_full_name}/contents"
            
            async with session.get(url) as response:
                if response.status == 200:
                    root_contents = await response.json()
                    await self._process_contents(session, repo_full_name, root_contents, contents)
        
        return contents
    
    async def _process_contents(self, session: aiohttp.ClientSession, repo_name: str, 
                              items: List[Dict], contents: Dict, path: str = ""):
        """Process repository contents recursively."""
        for item in items:
            item_info = {
                "name": item["name"],
                "path": item["path"],
                "type": item["type"],
                "size": item.get("size", 0),
                "extension": item["name"].split(".")[-1].lower() if "." in item["name"] else ""
            }
            
            if item["type"] == "file":
                contents["files"].append(item_info)
                
                # Special file handling
                if item["name"].lower() in ["readme.md", "readme.txt", "readme"]:
                    contents["readme"] = await self._get_file_content(session, repo_name, item["path"])
                elif item["name"] in ["requirements.txt", "environment.yml", "pyproject.toml"]:
                    contents["requirements"] = await self._get_file_content(session, repo_name, item["path"])
                elif item["name"].endswith(".ipynb"):
                    contents["notebooks"].append(item_info)
                    
            elif item["type"] == "dir" and not item["name"].startswith("."):
                contents["directories"].append(item_info)
                
                # Recursively process important directories
                if item["name"] in ["src", "lib", "notebooks", "examples", "models"]:
                    try:
                        subdir_url = f"{self.base_url}/repos/{repo_name}/contents/{item['path']}"
                        async with session.get(subdir_url) as response:
                            if response.status == 200:
                                subdir_contents = await response.json()
                                await self._process_contents(session, repo_name, subdir_contents, contents, item["path"])
                    except Exception as e:
                        self.logger.warning(f"Could not process subdirectory {item['path']}: {e}")
    
    async def _get_file_content(self, session: aiohttp.ClientSession, repo_name: str, 
                               file_path: str) -> Optional[str]:
        """Get content of a specific file."""
        try:
            url = f"{self.base_url}/repos/{repo_name}/contents/{file_path}"
            async with session.get(url) as response:
                if response.status == 200:
                    file_data = await response.json()
                    if file_data.get("encoding") == "base64":
                        content = base64.b64decode(file_data["content"]).decode("utf-8")
                        return content
        except Exception as e:
            self.logger.warning(f"Could not get file content for {file_path}: {e}")
        return None
    
    async def get_repository_metadata(self, repo_full_name: str) -> Optional[Dict]:
        """Get repository metadata."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            url = f"{self.base_url}/repos/{repo_full_name}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        return None


# framework_detector.py
"""
Framework and Technology Detection
"""
import re
from typing import Dict, List, Set, Any
import json

class FrameworkDetector:
    """Detect AI/ML frameworks and technologies in repositories."""
    
    def __init__(self):
        self.frameworks_db = {
            'ml_frameworks': {
                'tensorflow': ['tensorflow', 'tf.', 'import tensorflow'],
                'pytorch': ['torch', 'pytorch', 'import torch'],
                'scikit-learn': ['sklearn', 'scikit-learn', 'from sklearn'],
                'keras': ['keras', 'from keras'],
                'xgboost': ['xgboost', 'import xgb'],
                'lightgbm': ['lightgbm', 'import lgb'],
                'catboost': ['catboost'],
                'jax': ['import jax', 'jax.']
            },
            'llm_frameworks': {
                'transformers': ['transformers', 'from transformers'],
                'openai': ['openai', 'import openai'],
                'langchain': ['langchain', 'from langchain'],
                'llamaindex': ['llamaindex', 'llama_index'],
                'anthropic': ['anthropic', 'import anthropic'],
                'cohere': ['cohere'],
                'huggingface': ['huggingface_hub']
            },
            'deep_learning': {
                'tensorflow': ['tensorflow', 'tf.'],
                'pytorch': ['torch', 'pytorch'],
                'mxnet': ['mxnet'],
                'caffe': ['caffe'],
                'theano': ['theano'],
                'onnx': ['onnx']
            },
            'data_frameworks': {
                'pandas': ['pandas', 'import pd'],
                'numpy': ['numpy', 'import np'],
                'dask': ['dask'],
                'polars': ['polars'],
                'spark': ['pyspark', 'spark']
            },
            'computer_vision': {
                'opencv': ['cv2', 'opencv'],
                'pillow': ['PIL', 'from PIL'],
                'imageio': ['imageio'],
                'scikit-image': ['skimage']
            },
            'visualization': {
                'matplotlib': ['matplotlib', 'pyplot'],
                'plotly': ['plotly'],
                'seaborn': ['seaborn', 'import sns'],
                'bokeh': ['bokeh'],
                'altair': ['altair']
            }
        }
    
    async def detect_frameworks(self, repository_contents: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Detect frameworks from repository contents."""
        detected = {category: [] for category in self.frameworks_db.keys()}
        
        # Check requirements files
        if repository_contents.get('requirements'):
            requirements_frameworks = self._detect_from_requirements(
                repository_contents['requirements']
            )
            for category, frameworks in requirements_frameworks.items():
                detected[category].extend(frameworks)
        
        # Check Python files (simulate by checking file names and patterns)
        python_files = [f for f in repository_contents.get('files', []) if f['extension'] == 'py']
        
        for file_info in python_files[:20]:  # Limit processing
            file_frameworks = self._detect_from_filename_and_path(file_info)
            for category, frameworks in file_frameworks.items():
                detected[category].extend(frameworks)
        
        # Remove duplicates
        for category in detected:
            detected[category] = list({fw['name']: fw for fw in detected[category]}.values())
        
        return detected
    
    def _detect_from_requirements(self, requirements_content: str) -> Dict[str, List[Dict]]:
        """Detect frameworks from requirements file."""
        detected = {category: [] for category in self.frameworks_db.keys()}
        
        lines = requirements_content.lower().split('\n')
        
        for category, frameworks in self.frameworks_db.items():
            for framework_name, patterns in frameworks.items():
                for pattern in patterns:
                    if any(pattern.lower() in line for line in lines):
                        detected[category].append({
                            'name': framework_name,
                            'confidence': 0.9,
                            'source': 'requirements'
                        })
                        break
        
        return detected
    
    def _detect_from_filename_and_path(self, file_info: Dict) -> Dict[str, List[Dict]]:
        """Detect frameworks from file names and paths."""
        detected = {category: [] for category in self.frameworks_db.keys()}
        
        filename = file_info['name'].lower()
        path = file_info['path'].lower()
        
        # Check for framework indicators in filenames
        for category, frameworks in self.frameworks_db.items():
            for framework_name, patterns in frameworks.items():
                for pattern in patterns:
                    if (pattern.lower() in filename or 
                        pattern.lower() in path or
                        any(keyword in filename for keyword in pattern.lower().split())):
                        detected[category].append({
                            'name': framework_name,
                            'confidence': 0.7,
                            'source': 'filename'
                        })
                        break
        
        return detected


# data_models.py
"""
Data models for repository and portfolio data
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class RepositoryData:
    """Repository analysis data."""
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
    frameworks: Dict[str, List[Dict]]
    code_analysis: Dict[str, Any]
    ai_ml_relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass  
class PortfolioData:
    """Portfolio data structure."""
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
            'repositories': [repo.to_dict() if hasattr(repo, 'to_dict') else asdict(repo) 
                           for repo in self.repositories],
            'insights': self.insights,
            'categories': self.categories,
            'skills': self.skills,
            'highlights': self.highlights,
            'expertise_metrics': self.expertise_metrics
        }


# =================== COMPLETE WORKFLOW ===================

# workflow_orchestrator.py
"""
Complete AI Portfolio Scanner Workflow Orchestrator
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

class AIPortfolioWorkflow:
    """Complete workflow orchestrator for AI portfolio scanning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        self.results = {
            'repositories': [],
            'portfolio': None,
            'execution_time': 0,
            'errors': [],
            'statistics': {}
        }
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"portfolio_scan_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def execute_full_workflow(self, github_token: str, target: str, 
                                  target_type: str = 'user') -> Dict[str, Any]:
        """Execute the complete portfolio scanning workflow."""
        start_time = datetime.now()
        self.logger.info("Starting AI Portfolio Scanning Workflow")
        
        try:
            # Step 1: Initialize components
            self.logger.info("Initializing scanner components...")
            scanner = GitHubPortfolioScanner(github_token, None)
            
            # Step 2: Repository discovery and analysis
            self.logger.info(f"Scanning {target_type}: {target}")
            
            if target_type == 'user':
                repositories = await scanner.scan_user_repositories(target)
            elif target_type == 'repos':
                repo_urls = [url.strip() for url in target.split(',')]
                repositories = await scanner.scan_specific_repositories(repo_urls)
            else:
                raise ValueError(f"Invalid target_type: {target_type}")
            
            # Step 3: Build portfolio
            self.logger.info("Building portfolio from scanned repositories...")
            scanner.scanned_repos = repositories
            portfolio = scanner.build_portfolio()
            
            # Step 4: Generate reports
            self.logger.info("Generating portfolio reports...")
            output_path = await self._save_comprehensive_results(
                repositories, portfolio, scanner
            )
            
            # Step 5: Calculate statistics
            execution_time = (datetime.now() - start_time).total_seconds()
            statistics = self._calculate_workflow_statistics(
                repositories, portfolio, execution_time
            )
            
            self.results = {
                'repositories': repositories,
                'portfolio': portfolio,
                'execution_time': execution_time,
                'output_path': str(output_path),
                'statistics': statistics,
                'success': True
            }
            
            self.logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")
            return self.results
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            self.results.update({
                'success': False,
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            })
            return self.results
    
    async def _save_comprehensive_results(self, repositories: List[Any], 
                                        portfolio: Any, scanner: Any) -> Path:
        """Save comprehensive results with multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"ai_portfolio_scan_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Raw data export
        with open(output_dir / "repositories_raw.json", 'w') as f:
            json.dump([repo.to_dict() for repo in repositories], f, indent=2, default=str)
        
        # 2. Portfolio data
        with open(output_dir / "portfolio_data.json", 'w') as f:
            json.dump(portfolio.to_dict(), f, indent=2, default=str)
        
        # 3. HTML Report
        html_report = scanner.portfolio_builder.generate_html_report(portfolio)
        with open(output_dir / "portfolio_report.html", 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # 4. Markdown Summary  
        md_summary = scanner.portfolio_builder.generate_markdown_summary(portfolio)
        with open(output_dir / "portfolio_summary.md", 'w', encoding='utf-8') as f:
            f.write(md_summary)
        
        # 5. Skills Matrix
        skills_matrix = scanner.portfolio_builder.generate_skills_matrix(portfolio)
        with open(output_dir / "skills_matrix.json", 'w') as f:
            json.dump(skills_matrix, f, indent=2)
        
        # 6. Executive Summary
        exec_summary = self._generate_executive_summary(portfolio)
        with open(output_dir / "executive_summary.md", 'w') as f:
            f.write(exec_summary)
        
        # 7. Configuration used
        with open(output_dir / "scan_configuration.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Comprehensive results saved to: {output_dir}")
        return output_dir
    
    def _calculate_workflow_statistics(self, repositories: List[Any], 
                                     portfolio: Any, execution_time: float) -> Dict[str, Any]:
        """Calculate comprehensive workflow statistics."""
        return {
            'execution_time_seconds': execution_time,
            'repositories_scanned': len(repositories),
            'relevant_repositories': len([r for r in repositories if r.ai_ml_relevance_score >= 3.0]),
            'total_github_stars': sum(r.stars for r in repositories),
            'total_github_forks': sum(r.forks for r in repositories),
            'unique_frameworks_detected': len(portfolio.skills.get('ml_frameworks', {})),
            'programming_languages': len(portfolio.skills.get('programming_languages', {})),
            'portfolio_categories': len(portfolio.categories),
            'highlighted_projects': len(portfolio.highlights),
            'overall_expertise_score': portfolio.expertise_metrics.get('overall_score', 0),
            'scan_efficiency': len(repositories) / execution_time if execution_time > 0 else 0
        }
    
    def _generate_executive_summary(self, portfolio: Any) -> str:
        """Generate executive summary."""
        stats = self.results['statistics']
        
        summary = f"""# AI Portfolio Executive Summary

## Key Metrics
- **Repositories Analyzed**: {stats['repositories_scanned']}
- **AI/ML Relevant Projects**: {stats['relevant_repositories']} 
- **GitHub Stars**: {stats['total_github_stars']:,}
- **Community Forks**: {stats['total_github_forks']:,}
- **Overall Expertise Score**: {stats['overall_expertise_score']:.1f}/10

## Technical Profile
- **Frameworks Used**: {stats['unique_frameworks_detected']} different AI/ML frameworks
- **Programming Languages**: {stats['programming_languages']} languages
- **Domain Categories**: {stats['portfolio_categories']} specialized areas
- **Highlighted Projects**: {stats['highlighted_projects']} standout repositories

## Strengths
"""
        
        # Add strengths based on expertise metrics
        expertise = portfolio.expertise_metrics
        top_areas = sorted(expertise.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for area, score in top_areas:
            if score >= 7.0:
                area_name = area.replace('_', ' ').title()
                summary += f"- **{area_name}**: {score:.1f}/10 (Strong expertise)\n"
        
        summary += f"""
## Portfolio Highlights
"""
        
        for i, highlight in enumerate(portfolio.highlights[:3], 1):
            summary += f"""
### {i}. {highlight['name']}
- **Relevance**: {highlight['relevance_score']:.1f}/10
- **Community**: {highlight['stars']} stars, {highlight['forks']} forks  
- **Key Tech**: {', '.join(highlight['key_frameworks'][:3])}
- **Description**: {highlight['description'][:150]}...
"""
        
        summary += f"""
## Scan Efficiency
- **Execution Time**: {stats['execution_time_seconds']:.1f} seconds
- **Processing Rate**: {stats['scan_efficiency']:.1f} repos/second
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return summary


# =================== MAIN EXECUTION SCRIPT ===================

# main.py - Complete execution script
"""
AI Portfolio Scanner - Complete Execution Script
"""

async def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Portfolio Scanner - Complete Workflow')
    parser.add_argument('--token', required=True, help='GitHub API token')
    parser.add_argument('--user', help='GitHub username to scan')
    parser.add_argument('--repos', help='Comma-separated repository URLs')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-dir', default='portfolio_output', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'analysis': {
            'max_repos': 50,
            'include_forks': False,
            'min_stars': 0
        },
        'output': {
            'generate_html': True,
            'generate_markdown': True,
            'include_code_samples': True
        }
    }
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
    
    # Initialize workflow
    workflow = AIPortfolioWorkflow(config)
    
    try:
        if args.user:
            results = await workflow.execute_full_workflow(
                args.token, args.user, 'user'
            )
        elif args.repos:
            results = await workflow.execute_full_workflow(
                args.token, args.repos, 'repos'
            )
        else:
            print("Please specify either --user or --repos")
            sys.exit(1)
        
        if results['success']:
            print(f"✅ Portfolio scan completed successfully!")
            print(f"📁 Results saved to: {results['output_path']}")
            print(f"📊 Analyzed {results['statistics']['repositories_scanned']} repositories")
            print(f"⭐ Found {results['statistics']['relevant_repositories']} AI/ML projects")
            print(f"🏆 Overall expertise score: {results['statistics']['overall_expertise_score']:.1f}/10")
            print(f"⏱️ Completed in {results['execution_time']:.1f} seconds")
        else:
            print(f"❌ Portfolio scan failed: {results.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("🛑 Scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Import all required modules at runtime
    import os
    import sys
    from pathlib import Path
    
    # Add current directory to path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Import our modules
    from repo_analyzer import RepoAnalyzer
    from code_analyzer import CodeAnalyzer  
    from framework_detector import FrameworkDetector
    from portfolio_builder import PortfolioBuilder
    from data_models import RepositoryData, PortfolioData
    from llm_portfolio_scanner import GitHubPortfolioScanner
    
    # Run the main workflow
    asyncio.run(main())


# =================== DEPLOYMENT CONFIGURATION ===================

# requirements.txt
"""
aiohttp==3.8.5
asyncio==3.4.3
pyyaml==6.0
jinja2==3.1.2
pathlib==1.0.1
dataclasses==0.6
typing-extensions==4.7.1
"""

# config.yaml - Default configuration
"""
frameworks:
  ml_frameworks: ['tensorflow', 'pytorch', 'scikit-learn', 'keras', 'xgboost']
  llm_frameworks: ['transformers', 'openai', 'langchain', 'llamaindex', 'anthropic']
  dl_frameworks: ['torch', 'tensorflow', 'mxnet', 'caffe']
  data_frameworks: ['pandas', 'numpy', 'dask', 'polars']
  viz_frameworks: ['matplotlib', 'plotly', 'seaborn', 'bokeh']

analysis:
  max_repos: 100
  max_file_size: 1048576  # 1MB
  include_forks: false
  min_stars: 1
  exclude_archived: true

output:
  format: 'json'
  generate_html: true
  generate_markdown: true
  include_code_samples: true
  max_code_sample_lines: 50

logging:
  level: 'INFO'
  save_logs: true
"""

# Usage Instructions
"""
USAGE INSTRUCTIONS:

1. Install dependencies:
   pip install -r requirements.txt

2. Set up GitHub token:
   export GITHUB_TOKEN="your_github_token_here"

3. Scan a user's repositories:
   python main.py --token $GITHUB_TOKEN --user "username"

4. Scan specific repositories:
   python main.py --token $GITHUB_TOKEN --repos "owner/repo1,owner/repo2"

5. Use custom configuration:
   python main.py --token $GITHUB_TOKEN --user "username" --config config.yaml

6. Results will be saved in timestamped directory with:
   - portfolio_report.html (Interactive web report)
   - portfolio_summary.md (Markdown summary) 
   - portfolio_data.json (Raw data)
   - executive_summary.md (Executive overview)
   - skills_matrix.json (Detailed skills breakdown)
"""
