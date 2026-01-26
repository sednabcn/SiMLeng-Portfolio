# Portfolio Configuration Guide

## Features Added ✨

### 1. **Project Status Labels**
- 🟢 **CURRENT** - Projects updated in the last 30 days
- 🟡 **RECENT** - Projects updated in the last 6 months
- ⚪ **PAST** - Projects not updated in 6+ months

### 2. **Repository Blacklist**
Hide specific repositories from your portfolio

### 3. **Whitelist Mode** (Optional)
Only show specific repositories

---

## Quick Setup

### Method 1: Environment Variables (Quick & Easy)

Edit your workflow file (`.github/workflows/generate-portfolio.yml`):

```yaml
- name: Generate portfolio
  env:
    GITHUB_TOKEN: ${{ secrets.PORTFOLIO_SCANNER }}
    TARGET_USER: sednabcn
    INCLUDE_PRIVATE_REPOS: true
    # Add repos to hide (comma-separated)
    REPO_BLACKLIST: test-repo,old-project,scratch-notes
```

### Method 2: Configuration File (Recommended)

Create `.github/portfolio-config.yml`:

```yaml
# Repositories to hide
blacklist:
  - test-repo
  - scratch-work
  - old-experiments
  - private-notes
  - learning-sandbox

# Manual status overrides (optional)
status_overrides:
  my-ml-pipeline: current
  legacy-project: past

# Settings
min_relevance_score: 1.0
include_forks: false
include_private: true
```

---

## How to Use

### Option A: Blacklist Mode (Default)
**Hide specific repos**, show everything else

```yaml
blacklist:
  - repo-to-hide-1
  - repo-to-hide-2
  - test-project
```

### Option B: Whitelist Mode
**Only show specific repos**, hide everything else

```yaml
whitelist:
  - important-project-1
  - ml-pipeline
  - data-science-toolkit
```

⚠️ **Note**: If `whitelist` is defined, `blacklist` is ignored!

---

## Manual Status Labels

### Method 1: GitHub Topics (Easiest)
Add topics to your repos on GitHub:
- `current-project` → Shows as 🟢 CURRENT
- `past-project` or `completed` → Shows as ⚪ PAST

### Method 2: Config File
```yaml
status_overrides:
  my-active-project: current
  old-ml-model: past
  experiment-2024: recent
```

---

## Additional Settings

```yaml
# Minimum AI/ML relevance score (0-10)
min_relevance_score: 1.0

# Include forked repositories
include_forks: false

# Include private repositories
include_private: true

# Show "Private" badge on private repos
show_private_badge: true

# Group repos by status (current/recent/past)
group_by_status: true

# Sort order: score, stars, updated, or name
sort_by: score
```

---

## Common Scenarios

### Hide all test/experimental repos
```yaml
blacklist:
  - test-*  # Note: Exact name matching only for now
  - scratch-*
  - experiment-*
  - demo-*
```

### Only show your best projects
```yaml
whitelist:
  - ml-production-system
  - deep-learning-framework
  - data-pipeline-automation
  
min_relevance_score: 5.0  # Only high-scoring projects
```

### Highlight current work
```yaml
status_overrides:
  ai-chatbot: current
  nlp-service: current
  old-classifier: past
```

---

## Files to Update

### 1. Update `data_models.py`
Add `project_status` field to `RepositoryData`

### 2. Update `repo_analyzer.py`
Fix private repo fetching (use `/user/repos`)

### 3. Create `config_loader.py`
Add configuration file support

### 4. Update `generate_portfolio.py`
- Import `PortfolioConfig`
- Use config for blacklist/whitelist
- Pass config to `determine_project_status()`

### 5. Create `.github/portfolio-config.yml`
Your configuration file

### 6. Update workflow
Add `REPO_BLACKLIST` environment variable

---

## Testing

Run locally to test:
```bash
export GITHUB_TOKEN="your_token"
export TARGET_USER="sednabcn"
export INCLUDE_PRIVATE_REPOS="true"
export REPO_BLACKLIST="test-repo,old-project"

python .github/scripts/generate_portfolio.py
```

Check the output for:
- ✓ Repos correctly included/excluded
- 🟢 Status labels appearing correctly
- Blacklisted count in summary

---

## Troubleshooting

### Private repos not showing?
1. Check token has `repo` scope
2. Verify `repo_analyzer.py` uses `/user/repos` endpoint
3. Set `INCLUDE_PRIVATE_REPOS: true`

### Blacklist not working?
1. Use exact repo names (not full names)
2. Check spelling matches GitHub exactly
3. Check config file is being loaded

###