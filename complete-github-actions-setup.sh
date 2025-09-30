#!/bin/bash
# Complete GitHub Actions Setup for SiMLeng-Portfolio
# Run this in: ~/Downloads/GITHUB/SiMLeng-Portfolio-master

set -e

echo "Setting up GitHub Actions for AI Portfolio Scanner..."

# Create directory structure
mkdir -p .github/workflows
mkdir -p .github/scripts
mkdir -p src
mkdir -p docs

# Move existing files to proper locations
echo "Organizing existing files..."
mv code-analyzer-module.py src/code_analyzer.py 2>/dev/null || true
mv portfolio-builder-module.py src/portfolio_builder.py 2>/dev/null || true
mv llm-portfolio-scanner.py src/scanner_base.py 2>/dev/null || true

# Create missing data models
cat > src/data_models.py << 'EOF'
"""Data models for portfolio scanner"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional

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
    code_analysis: Dict[str, Any]
    ai_ml_relevance_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class PortfolioData:
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
            'repositories': [r.to_dict() for r in self.repositories],
            'insights': self.insights,
            'categories': self.categories,
            'skills': self.skills,
            'highlights': self.highlights,
            'expertise_metrics': self.expertise_metrics
        }
EOF

# Create simple repo analyzer (wrapper)
cat > src/repo_analyzer.py << 'EOF'
"""Simple repo analyzer using GitHub API"""
import aiohttp
import asyncio
from typing import Dict, List, Optional

class RepoAnalyzer:
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    async def get_user_repositories(self, username: str, max_repos: int = 100, 
                                  include_forks: bool = False) -> List[Dict]:
        repos = []
        page = 1
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            while len(repos) < max_repos:
                url = f"{self.base_url}/users/{username}/repos"
                params = {"page": page, "per_page": 100, "sort": "updated"}
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        break
                    batch = await response.json()
                    if not batch:
                        break
                    
                    for repo in batch:
                        if include_forks or not repo.get('fork', False):
                            repos.append(repo)
                    page += 1
        
        return repos[:max_repos]
EOF

# Create framework detector
cat > src/framework_detector.py << 'EOF'
"""Framework detection from repository contents"""
from typing import Dict, List, Any

class FrameworkDetector:
    def __init__(self):
        self.frameworks = {
            'ml_frameworks': ['tensorflow', 'pytorch', 'scikit-learn', 'keras'],
            'llm_frameworks': ['transformers', 'openai', 'langchain', 'anthropic'],
            'data_frameworks': ['pandas', 'numpy', 'dask']
        }
    
    async def detect_frameworks(self, repo_data: Dict) -> Dict[str, List[str]]:
        detected = {k: [] for k in self.frameworks.keys()}
        
        description = (repo_data.get('description') or '').lower()
        topics = [t.lower() for t in repo_data.get('topics', [])]
        text = description + ' ' + ' '.join(topics)
        
        for category, frameworks in self.frameworks.items():
            for framework in frameworks:
                if framework in text:
                    detected[category].append(framework)
        
        return detected
EOF

touch src/__init__.py

# Create the main GitHub Actions script
cat > .github/scripts/generate_portfolio.py << 'EOF'
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
EOF

chmod +x .github/scripts/generate_portfolio.py

# Create GitHub Actions workflow
cat > .github/workflows/generate-portfolio.yml << 'EOF'
name: Generate Portfolio

on:
  push:
    branches: [ master, main ]
  workflow_dispatch:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  generate:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install aiohttp pyyaml jinja2
    
    - name: Generate portfolio
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        TARGET_USER: sednabcn
      run: |
        python .github/scripts/generate_portfolio.py
    
    - name: Commit and push
      run: |
        git config user.name "github-actions[bot]"
        git config user.email "github-actions[bot]@users.noreply.github.com"
        git add docs/
        git diff --quiet && git diff --staged --quiet || (git commit -m "Update portfolio" && git push)
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
aiohttp>=3.8.5
pyyaml>=6.0
jinja2>=3.1.2
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.log
.env
*.token
venv/
EOF

# Create README update
cat > WORKFLOW_SETUP.md << 'EOF'
# GitHub Actions Portfolio Setup

## What This Does

The GitHub Actions workflow automatically:
1. Scans your repositories (sednabcn)
2. Analyzes AI/ML projects
3. Generates portfolio HTML and markdown
4. Commits results to docs/ folder
5. Updates every week (Sunday) or on push

## Files Created

```
.github/
├── workflows/
│   └── generate-portfolio.yml  # Main workflow
└── scripts/
    └── generate_portfolio.py   # Scanner script

src/
├── code_analyzer.py
├── portfolio_builder.py
├── data_models.py
├── repo_analyzer.py
└── framework_detector.py

docs/                           # Generated portfolio
├── index.html
├── README.md
└── portfolio.json
```

## Setup Steps

1. Commit everything:
   ```bash
   git add .
   git commit -m "Setup GitHub Actions portfolio scanner"
   git push origin master
   ```

2. Enable GitHub Pages:
   - Go to repo Settings → Pages
   - Source: Deploy from a branch
   - Branch: master
   - Folder: /docs
   - Save

3. View your portfolio at:
   https://sednabcn.github.io/SiMLeng-Portfolio/

## Manual Trigger

Go to Actions tab → Generate Portfolio → Run workflow

## What Gets Generated

- **docs/index.html** - Interactive portfolio
- **docs/README.md** - Markdown summary
- **docs/portfolio.json** - Raw data
EOF

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Setup GitHub Actions portfolio scanner'"
echo "3. git push origin master"
echo "4. Go to repo Settings → Pages → Enable from docs/ folder"
echo "5. Visit: https://sednabcn.github.io/SiMLeng-Portfolio/"
echo ""
echo "Read WORKFLOW_SETUP.md for details"
