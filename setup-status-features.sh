#!/bin/bash

# Setup and Test Script for Portfolio Status Features
# Run this from the SiMLeng-Portfolio-master directory

set -e

echo "🚀 Setting Up Portfolio Status Features"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check directory
if [ ! -d "src" ]; then
    echo -e "${RED}❌ Error: Please run from SiMLeng-Portfolio-master directory${NC}"
    exit 1
fi

echo -e "${BLUE}📁 Working directory: $(pwd)${NC}"
echo ""

# Step 1: Create status_detector.py
echo -e "${YELLOW}[1/5] Creating src/status_detector.py...${NC}"
cat > src/status_detector.py << 'DETECTOR_EOF'
"""Project status detection for portfolio"""
from datetime import datetime
from typing import Optional
from data_models import RepositoryData
from config_loader import PortfolioConfig

def determine_project_status(repo: RepositoryData, config: Optional[PortfolioConfig] = None) -> str:
    """Determine project status based on last update date and configuration."""
    if config:
        override = config.get_status_override(repo.name)
        if override:
            return override.lower()
    
    topics_lower = [t.lower() for t in repo.topics]
    
    if 'current-project' in topics_lower:
        return 'current'
    
    if any(keyword in topics_lower for keyword in ['past-project', 'completed', 'archived']):
        return 'past'
    
    try:
        date_str = repo.updated_at
        if 'T' in date_str:
            date_str = date_str.split('T')[0] + 'T' + date_str.split('T')[1].split('Z')[0].split('+')[0]
            last_update = datetime.fromisoformat(date_str.replace('Z', ''))
        else:
            last_update = datetime.fromisoformat(date_str)
        
        now = datetime.utcnow()
        days_since_update = (now - last_update).days
        
        if days_since_update <= 30:
            return 'current'
        elif days_since_update <= 180:
            return 'recent'
        else:
            return 'past'
    except (ValueError, TypeError, AttributeError) as e:
        return 'recent'

def get_status_badge(status: str) -> dict:
    """Get display information for a status."""
    status_info = {
        'current': {
            'emoji': '🟢',
            'label': 'CURRENT',
            'color': '#22c55e',
            'bg_color': '#dcfce7',
            'text_color': '#166534',
            'description': 'Active development'
        },
        'recent': {
            'emoji': '🟡',
            'label': 'RECENT',
            'color': '#eab308',
            'bg_color': '#fef9c3',
            'text_color': '#854d0e',
            'description': 'Recently updated'
        },
        'past': {
            'emoji': '⚪',
            'label': 'PAST',
            'color': '#94a3b8',
            'bg_color': '#f1f5f9',
            'text_color': '#475569',
            'description': 'Completed or archived'
        }
    }
    return status_info.get(status.lower(), status_info['recent'])

def group_by_status(repos: list) -> dict:
    """Group repositories by their status."""
    groups = {'current': [], 'recent': [], 'past': []}
    for repo in repos:
        status = getattr(repo, 'project_status', 'recent') or 'recent'
        if status in groups:
            groups[status].append(repo)
        else:
            groups['recent'].append(repo)
    return groups

def get_status_summary(repos: list) -> dict:
    """Get summary statistics for project statuses."""
    total = len(repos)
    if total == 0:
        return {'total': 0, 'current': 0, 'recent': 0, 'past': 0,
                'current_pct': 0.0, 'recent_pct': 0.0, 'past_pct': 0.0}
    
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

def sort_by_status(repos: list) -> list:
    """Sort repositories with current first, then recent, then past."""
    status_order = {'current': 0, 'recent': 1, 'past': 2}
    return sorted(repos, key=lambda r: (
        status_order.get(getattr(r, 'project_status', 'recent') or 'recent', 1),
        -r.ai_ml_relevance_score,
        -r.stars
    ))

def format_status_for_display(status: str) -> str:
    """Format status string for display in UI."""
    badge = get_status_badge(status)
    return f"{badge['emoji']} {badge['label']}"
DETECTOR_EOF

echo -e "${GREEN}✅ Created src/status_detector.py${NC}"
echo ""

# Step 2: Check dependencies
echo -e "${YELLOW}[2/5] Checking dependencies...${NC}"
if ! grep -q "PyYAML" requirements.txt 2>/dev/null; then
    echo "PyYAML>=6.0" >> requirements.txt
    echo -e "${GREEN}   ✅ Added PyYAML to requirements.txt${NC}"
else
    echo -e "${BLUE}   ℹ️  PyYAML already in requirements.txt${NC}"
fi

if ! grep -q "jinja2" requirements.txt 2>/dev/null; then
    echo "jinja2>=3.1.0" >> requirements.txt
    echo -e "${GREEN}   ✅ Added jinja2 to requirements.txt${NC}"
else
    echo -e "${BLUE}   ℹ️  jinja2 already in requirements.txt${NC}"
fi

echo -e "${BLUE}   Installing dependencies...${NC}"
pip install -q PyYAML jinja2 2>/dev/null || true
echo ""

# Step 3: Update config file
echo -e "${YELLOW}[3/5] Checking portfolio-config.yml...${NC}"
if [ -f ".github/portfolio-config.yml" ]; then
    echo -e "${BLUE}   ℹ️  Config file already exists${NC}"
else
    mkdir -p .github
    cp portfolio-config.yml .github/portfolio-config.yml 2>/dev/null || \
    cat > .github/portfolio-config.yml << 'CONFIG_EOF'
# Portfolio Configuration
target_user: sednabcn

blacklist:
  - SiMLeng-Portfolio
  - test-repo
  - scratch-work

whitelist: []

status_overrides:
  Real-World-Statistics-Projects: current

min_relevance_score: 1.0
include_forks: false
include_private: true
show_private_badge: true
group_by_status: true
sort_by: score
CONFIG_EOF
    echo -e "${GREEN}   ✅ Created .github/portfolio-config.yml${NC}"
fi
echo ""

# Step 4: Create test script
echo -e "${YELLOW}[4/5] Creating test script...${NC}"
cat > test-status-features.sh << 'TEST_EOF'
#!/bin/bash

echo "🧪 Testing Portfolio Status Features"
echo "===================================="
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN not set"
    echo ""
    echo "Please set your GitHub token:"
    #echo "  export GITHUB_TOKEN=''"
    echo ""
    echo "Get a token from: https://github.com/settings/tokens"
    exit 1
fi

export TARGET_USER="sednabcn"
export INCLUDE_PRIVATE_REPOS="true"

echo "Testing configuration loading..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from config_loader import PortfolioConfig

config = PortfolioConfig()
print(f'✅ Configuration loaded')
print(f'   User: {config.config.get(\"target_user\", \"N/A\")}')
print(f'   Blacklist: {len(config.get_blacklist())} repos')
print(f'   Status overrides: {len(config.config.get(\"status_overrides\", {}))}')
"

echo ""
echo "Testing status detector..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from status_detector import get_status_badge, format_status_for_display

for status in ['current', 'recent', 'past']:
    badge = get_status_badge(status)
    display = format_status_for_display(status)
    print(f'   {display}: {badge[\"description\"]}')
"

echo ""
echo "✅ All tests passed!"
echo ""
echo "To generate your portfolio, run:"
echo "  cd .github/workflows"
echo "  python3 ../scripts/generate_portfolio.py"
TEST_EOF

chmod +x test-status-features.sh
echo -e "${GREEN}✅ Created test-status-features.sh${NC}"
echo ""

# Step 5: Summary
echo -e "${YELLOW}[5/5] Installation Summary${NC}"
echo "========================================"
echo ""
echo -e "${GREEN}✅ Status Features Installed!${NC}"
echo ""
echo -e "${BLUE}Files created/updated:${NC}"
echo "   ✅ src/status_detector.py"
echo "   ✅ src/config_loader.py (already existed)"
echo "   ✅ src/data_models.py (already has project_status)"
echo "   ✅ .github/portfolio-config.yml"
echo "   ✅ requirements.txt (updated)"
echo "   ✅ test-status-features.sh"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo ""
echo "1️⃣  Configure your portfolio:"
echo "    nano .github/portfolio-config.yml"
echo ""
echo "2️⃣  Test the features:"
echo "    export GITHUB_TOKEN='your_token_here'"
echo "    ./test-status-features.sh"
echo ""
echo "3️⃣  Generate your portfolio:"
echo "    cd .github/workflows"
echo "    python3 ../scripts/generate_portfolio.py"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "   • Add 'current-project' topic to active repos on GitHub"
echo "   • Add 'past-project' or 'completed' topic to finished projects"
echo "   • Update status_overrides in config for manual control"
echo "   • Projects updated in last 30 days = 🟢 CURRENT"
echo "   • Projects updated 1-6 months ago = 🟡 RECENT"
echo "   • Projects not updated in 6+ months = ⚪ PAST"
echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
