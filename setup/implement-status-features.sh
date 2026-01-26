#!/bin/bash

# Implementation script for CURRENT/RECENT/PAST status features
# This script sets up all required files and configurations

set -e  # Exit on error

echo "🚀 Implementing Portfolio Status Features..."
echo "=============================================="
echo ""

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "src" ]; then
    echo -e "${RED}❌ Error: Please run this script from the SiMLeng-Portfolio-master directory${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Current directory: $(pwd)${NC}"
echo ""

# Step 1: Create config_loader.py
echo -e "${YELLOW}[1/6] Creating config_loader.py...${NC}"
cat > src/config_loader.py << 'EOF'
"""Configuration loader for portfolio generation."""
import os
import yaml
from typing import Optional
from pathlib import Path
from data_models import PortfolioConfig

def load_config(config_path: Optional[str] = None) -> PortfolioConfig:
    """Load portfolio configuration from file or environment variables."""
    config_data = {}
    
    # Try to load from YAML file
    if config_path and Path(config_path).exists():
        print(f"📄 Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        default_path = Path('.github/portfolio-config.yml')
        if default_path.exists():
            print(f"📄 Loading configuration from {default_path}")
            with open(default_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
    
    target_user = os.getenv('TARGET_USER', config_data.get('target_user', ''))
    if not target_user:
        raise ValueError("TARGET_USER must be set")
    
    blacklist = []
    env_blacklist = os.getenv('REPO_BLACKLIST', '')
    if env_blacklist:
        blacklist = [repo.strip() for repo in env_blacklist.split(',') if repo.strip()]
    else:
        blacklist = config_data.get('blacklist', [])
    
    whitelist = config_data.get('whitelist', [])
    status_overrides = config_data.get('status_overrides', {})
    
    include_private = os.getenv('INCLUDE_PRIVATE_REPOS', 
                                str(config_data.get('include_private', True))).lower() == 'true'
    include_forks = config_data.get('include_forks', False)
    min_relevance_score = float(config_data.get('min_relevance_score', 1.0))
    show_private_badge = config_data.get('show_private_badge', True)
    group_by_status = config_data.get('group_by_status', True)
    sort_by = config_data.get('sort_by', 'score')
    
    config = PortfolioConfig(
        target_user=target_user,
        include_private=include_private,
        include_forks=include_forks,
        min_relevance_score=min_relevance_score,
        blacklist=blacklist,
        whitelist=whitelist,
        status_overrides=status_overrides,
        show_private_badge=show_private_badge,
        group_by_status=group_by_status,
        sort_by=sort_by
    )
    
    print("\n⚙️  Portfolio Configuration:")
    print(f"   Target User: {config.target_user}")
    print(f"   Include Private: {config.include_private}")
    if config.whitelist:
        print(f"   🎯 Whitelist Mode: {len(config.whitelist)} repos")
    elif config.blacklist:
        print(f"   🚫 Blacklist: {len(config.blacklist)} repos")
    print()
    
    return config
EOF

echo -e "${GREEN}✅ Created src/config_loader.py${NC}"
echo ""

# Step 2: Create status_detector.py
echo -e "${YELLOW}[2/6] Creating status_detector.py...${NC}"
cat > src/status_detector.py << 'EOF'
"""Project status detection for portfolio."""
from datetime import datetime, timedelta
from typing import Optional
from data_models import RepositoryData, PortfolioConfig

def determine_project_status(repo: RepositoryData, config: Optional[PortfolioConfig] = None) -> str:
    """Determine project status based on last update date and configuration."""
    if config:
        override = config.get_status_override(repo.name)
        if override:
            return override.lower()
    
    topics_lower = [t.lower() for t in repo.topics]
    
    if 'current-project' in topics_lower:
        return 'current'
    
    if 'past-project' in topics_lower or 'completed' in topics_lower or 'archived' in topics_lower:
        return 'past'
    
    try:
        date_str = repo.pushed_at or repo.updated_at
        last_update = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.utcnow()
        days_since_update = (now - last_update).days
        
        if days_since_update <= 30:
            return 'current'
        elif days_since_update <= 180:
            return 'recent'
        else:
            return 'past'
    except (ValueError, TypeError):
        return 'recent'

def get_status_badge(status: str) -> dict:
    """Get display information for a status."""
    status_info = {
        'current': {'emoji': '🟢', 'label': 'CURRENT', 'color': '#22c55e'},
        'recent': {'emoji': '🟡', 'label': 'RECENT', 'color': '#eab308'},
        'past': {'emoji': '⚪', 'label': 'PAST', 'color': '#94a3b8'}
    }
    return status_info.get(status, status_info['recent'])

def group_by_status(repos: list) -> dict:
    """Group repositories by their status."""
    groups = {'current': [], 'recent': [], 'past': []}
    for repo in repos:
        status = repo.project_status or 'recent'
        groups[status].append(repo)
    return groups

def get_status_summary(repos: list) -> dict:
    """Get summary statistics for project statuses."""
    total = len(repos)
    if total == 0:
        return {'total': 0, 'current': 0, 'recent': 0, 'past': 0}
    
    groups = group_by_status(repos)
    return {
        'total': total,
        'current': len(groups['current']),
        'recent': len(groups['recent']),
        'past': len(groups['past']),
        'current_pct': round(len(groups['current']) / total * 100, 1),
        'recent_pct': round(len(groups['recent']) / total * 100, 1),
        'past_pct': round(len(groups['past']) / total * 100, 1)
    }

def sort_by_status(repos: list):
    """Sort repositories with current first, then recent, then past."""
    status_order = {'current': 0, 'recent': 1, 'past': 2}
    return sorted(repos, key=lambda r: (
        status_order.get(r.project_status or 'recent', 1),
        -r.relevance_score,
        -r.stargazers_count
    ))
EOF

echo -e "${GREEN}✅ Created src/status_detector.py${NC}"
echo ""

# Step 3: Update data_models.py (add project_status field)
echo -e "${YELLOW}[3/6] Updating data_models.py...${NC}"

# Check if project_status already exists
if grep -q "project_status" src/data_models.py; then
    echo -e "${BLUE}   ℹ️  project_status field already exists in data_models.py${NC}"
else
    # Backup original
    cp src/data_models.py src/data_models.py.backup
    
    # Add project_status field before the to_dict method
    sed -i '/def to_dict/i\    \n    # Project status (NEW)\n    project_status: Optional[str] = None  # '\''current'\'', '\''recent'\'', or '\''past'\''' src/data_models.py
    
    # Add project_status to to_dict return
    sed -i "/'key_features': self.key_features,/a\            'project_status': self.project_status  # NEW" src/data_models.py
    
    echo -e "${GREEN}   ✅ Added project_status field${NC}"
fi

echo ""

# Step 4: Create portfolio configuration file
echo -e "${YELLOW}[4/6] Creating .github/portfolio-config.yml...${NC}"
mkdir -p .github
cat > .github/portfolio-config.yml << 'EOF'
# Portfolio Configuration
target_user: sednabcn

# Repositories to hide (blacklist mode)
blacklist:
  - test-repo
  - scratch-work
  # Add more repos to hide here

# Manual status overrides
status_overrides:
  Real-World-Statistics-Projects: current
  SiMLeng-Portfolio-master: current

# Settings
min_relevance_score: 1.0
include_forks: false
include_private: true
show_private_badge: true
group_by_status: true
sort_by: score
EOF

echo -e "${GREEN}✅ Created .github/portfolio-config.yml${NC}"
echo ""

# Step 5: Install PyYAML if not already installed
echo -e "${YELLOW}[5/6] Checking dependencies...${NC}"
if ! grep -q "PyYAML" requirements.txt 2>/dev/null; then
    echo "PyYAML>=6.0" >> requirements.txt
    echo -e "${GREEN}   ✅ Added PyYAML to requirements.txt${NC}"
else
    echo -e "${BLUE}   ℹ️  PyYAML already in requirements.txt${NC}"
fi

echo -e "${BLUE}   Installing/upgrading dependencies...${NC}"
pip install --upgrade PyYAML >/dev/null 2>&1 || true
echo ""

# Step 6: Create test script
echo -e "${YELLOW}[6/6] Creating test script...${NC}"
cat > test-status-features.sh << 'TESTEOF'
#!/bin/bash

# Test the status features locally

echo "🧪 Testing Portfolio Status Features..."
echo ""

# Check required environment variables
if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN not set"
    echo "   Export your GitHub token:"
    echo "   export GITHUB_TOKEN='your_token_here'"
    exit 1
fi

export TARGET_USER="sednabcn"
export INCLUDE_PRIVATE_REPOS="true"

echo "📊 Running portfolio generation..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from repo_analyzer import RepositoryAnalyzer
from config_loader import load_config
from status_detector import determine_project_status, get_status_summary

config = load_config()
print(f'Configuration loaded for: {config.target_user}')
print(f'Status overrides: {config.status_overrides}')
"

echo ""
echo "✅ Basic test complete!"
echo ""
echo "To run full portfolio generation:"
echo "   python3 src/portfolio_builder.py"
TESTEOF

chmod +x test-status-features.sh
echo -e "${GREEN}✅ Created test-status-features.sh${NC}"
echo ""

# Summary
echo "=============================================="
echo -e "${GREEN}✨ Implementation Complete!${NC}"
echo "=============================================="
echo ""
echo -e "${BLUE}📋 What was created:${NC}"
echo "   ✅ src/config_loader.py"
echo "   ✅ src/status_detector.py"
echo "   ✅ .github/portfolio-config.yml"
echo "   ✅ Updated src/data_models.py"
echo "   ✅ Updated requirements.txt"
echo "   ✅ test-status-features.sh"
echo ""
echo -e "${BLUE}📌 Next Steps:${NC}"
echo "   1. Review .github/portfolio-config.yml"
echo "   2. Add your repository blacklist/whitelist"
echo "   3. Set status overrides for specific projects"
echo "   4. Run: ./test-status-features.sh"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   • Add 'current-project' topic to GitHub repos → Auto-labeled as CURRENT"
echo "   • Add 'past-project' or 'completed' topic → Auto-labeled as PAST"
echo "   • Projects updated in last 30 days → Auto-labeled as CURRENT"
echo "   • Projects updated 1-6 months ago → Auto-labeled as RECENT"
echo "   • Projects not updated in 6+ months → Auto-labeled as PAST"
echo ""
echo -e "${GREEN}🎉 Ready to use!${NC}"
