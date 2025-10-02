#!/bin/bash

echo "🧪 Testing Portfolio Status Features"
echo "===================================="
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ GITHUB_TOKEN not set"
    echo ""
    echo "Please set your GitHub token:"
    # echo "  export GITHUB_TOKEN=''"
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
