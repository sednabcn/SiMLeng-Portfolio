# Quick Start Guide: Portfolio Status Features

## 🚀 Installation (3 Minutes)

### Step 1: Run Setup Script

```bash
cd ~/Downloads/GITHUB/SiMLeng-Portfolio-master
chmod +x setup-status-features.sh
./setup-status-features.sh
```

This creates:
- ✅ `src/status_detector.py` - Status detection logic
- ✅ `.github/portfolio-config.yml` - Your configuration
- ✅ `test-status-features.sh` - Test script
- ✅ Updated `requirements.txt` with PyYAML

---

## ⚙️ Configuration

### Edit `.github/portfolio-config.yml`:

```yaml
# Your GitHub username
target_user: sednabcn

# Hide specific repositories
blacklist:
  - test-repo
  - scratch-work
  - SiMLeng-Portfolio  # Hide this portfolio repo itself

# Manual status overrides (optional)
status_overrides:
  Real-World-Statistics-Projects: current
  my-active-ml-project: current
  old-experiment: past

# Settings
min_relevance_score: 1.0
include_private: true
group_by_status: true
```

---

## 🧪 Test

```bash
export GITHUB_TOKEN="ghp_your_token_here"
./test-status-features.sh
```

---

## 🎯 Usage

### Option 1: Generate Locally

```bash
cd .github/workflows
export GITHUB_TOKEN="ghp_your_token_here"
export TARGET_USER="sednabcn"
export INCLUDE_PRIVATE_REPOS="true"
python3 ../scripts/generate_portfolio.py
```

Output: `docs/index.html`

### Option 2: Use GitHub Actions

The workflow will automatically run when you push to main branch.

---

## 📊 Status Labels Explained

| Status | When Applied | Badge |
|--------|-------------|-------|
| 🟢 **CURRENT** | Updated in last 30 days | Green |
| 🟡 **RECENT** | Updated 1-6 months ago | Yellow |
| ⚪ **PAST** | Not updated in 6+ months | Gray |

### Override with GitHub Topics

Add these topics to your repos:
- `current-project` → Forces 🟢 CURRENT
- `past-project` or `completed` → Forces ⚪ PAST

### Override with Config

```yaml
status_overrides:
  my-repo: current  # Force this repo to show as CURRENT
```

---

## 🎨 Customization

### Change Status Timeframes

Edit `src/status_detector.py`:

```python
if days_since_update <= 30:      # Change to 60 for 2 months
    return 'current'
elif days_since_update <= 180:   # Change to 365 for 1 year
    return 'recent'
```

### Hide/Show Projects

**Blacklist Mode** (default):
```yaml
blacklist:
  - repo-to-hide-1
  - repo-to-hide-2
```

**Whitelist Mode** (only show specific repos):
```yaml
whitelist:
  - important-project-1
  - ml-pipeline
  - production-system
```

---

## 🔧 Troubleshooting

### Issue: "PyYAML not found"
```bash
pip install PyYAML
```

### Issue: Status not showing
1. Check config file exists: `.github/portfolio-config.yml`
2. Verify repo names match exactly (case-sensitive)
3. Clear cache and regenerate

### Issue: Private repos not showing
1. Check token has `repo` scope
2. Set `include_private: true` in config
3. Verify `INCLUDE_PRIVATE_REPOS=true` env var

---

## 📁 File Structure

```
SiMLeng-Portfolio-master/
├── src/
│   ├── status_detector.py       ✨ NEW
│   ├── config_loader.py         ✅ Already exists
│   ├── data_models.py           ✅ Already has project_status
│   ├── portfolio_builder.py     
│   └── generate_portfolio.py    
├── .github/
│   ├── portfolio-config.yml     ✨ NEW
│   └── workflows/
│       └── generate-portfolio.yml
├── docs/
│   ├── index.html              ← Generated output
│   └── portfolio.json
├── requirements.txt            ✅ Updated with PyYAML
└── test-status-features.sh     ✨ NEW
```

---

## 🎯 Examples

### Example 1: Include Statistics Project

Create GitHub repo first:
```bash
cd ~/Downloads/GITHUB/Real-World-Statistics-Projects
git init
git add .
git commit -m "Initial commit"
gh repo create Real-World-Statistics-Projects --public --source=. --push
```

Then add to config:
```yaml
status_overrides:
  Real-World-Statistics-Projects: current
```

### Example 2: Hide Portfolio Repo

```yaml
blacklist:
  - SiMLeng-Portfolio-master
  - sednabcn.github.io
```

### Example 3: Only Show Best Work

```yaml
whitelist:
  - Real-World-Statistics-Projects
  - best-ml-project
  - production-app
  
min_relevance_score: 5.0
```

---

## 🚨 Common Mistakes

❌ **Wrong**: Repo name includes owner
```yaml
blacklist:
  - sednabcn/test-repo  # DON'T DO THIS
```

✅ **Correct**: Just the repo name
```yaml
blacklist:
  - test-repo  # CORRECT
```

---

❌ **Wrong**: Status value capitalized
```yaml
status_overrides:
  my-repo: CURRENT  # DON'T DO THIS
```

✅ **Correct**: Lowercase status
```yaml
status_overrides:
  my-repo: current  # CORRECT
```

---

## 📞 Need Help?

1. Run the test script: `./test-status-features.sh`
2. Check logs in the output
3. Verify config file syntax (YAML is sensitive to indentation)

---

## ✨ Features Summary

✅ **Automatic Status Detection** - Based on last update date  
✅ **Manual Overrides** - Via config file or GitHub topics  
✅ **Blacklist/Whitelist** - Control which repos appear  
✅ **Status Badges** - Visual indicators in HTML  
✅ **Grouped Display** - Projects organized by status  
✅ **Flexible Sorting** - By status, score, stars, or name  

---

**Ready to generate your portfolio?** 🎉

```bash
export GITHUB_TOKEN="ghp_your_token_here"
cd .github/workflows
python3 ../scripts/generate_portfolio.py
```

Then open `docs/index.html` in your browser!