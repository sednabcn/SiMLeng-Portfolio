# Configuration Guide for AI Portfolio Scanner

## 📋 Overview

The `config.yaml` file controls every aspect of how your AI portfolio is scanned, analyzed, and presented. This guide explains all configuration options.

---

## 🎯 Quick Configuration Tasks

### Common Tasks

**Exclude specific repositories:**
```yaml
blacklist:
  - my-test-repo
  - old-project
```

**Only include specific repositories:**
```yaml
whitelist:
  - LLM-HypatiaX
  - DeepSeek-R1
  - deep-learning
```

**Mark a project as current:**
```yaml
status_overrides:
  LLM-HypatiaX: current
  old-project: past
```

**Scan more repositories:**
```yaml
analysis:
  max_repos: 100  # Default is 50
```

**Include your forks:**
```yaml
analysis:
  include_forks: true  # Default is false
```

---

## 🔧 Configuration Sections

### 1. Framework Detection (`frameworks`)

Defines which AI/ML frameworks to detect in your code.

```yaml
frameworks:
  ml_frameworks:      # Classical ML
  llm_frameworks:     # Large Language Models
  dl_frameworks:      # Deep Learning
  data_frameworks:    # Data processing
  viz_frameworks:     # Visualization
  nlp_frameworks:     # NLP-specific
  cv_frameworks:      # Computer Vision
  rl_frameworks:      # Reinforcement Learning
```

**Add custom framework:**
```yaml
frameworks:
  ml_frameworks:
    - tensorflow
    - pytorch
    - your-custom-framework  # Add here
```

---

### 2. Analysis Configuration (`analysis`)

Controls how repositories are scanned and analyzed.

#### Key Settings:

**`max_repos`** (default: 50)
- Maximum number of repositories to scan
- Higher values = longer scan time
- Recommended: 50-100 for personal portfolios

**`max_file_size`** (default: 1048576 = 1 MB)
- Files larger than this are skipped
- Prevents scanning large binary files
- Adjust if you have large code files

**`include_forks`** (default: false)
- Include repositories you've forked
- Set to `true` if you've made significant contributions to forks

**`min_stars`** (default: 0)
- Only scan repos with at least this many stars
- Useful for highlighting popular projects

**`exclude_archived`** (default: true)
- Skip archived repositories
- Set to `false` to include historical work

**`code_extensions`**
- File types to analyze
- Add custom extensions if needed:
```yaml
code_extensions:
  - .py
  - .ipynb
  - .scala  # Add Scala
  - .go     # Add Go
```

**`ignore_patterns`**
- Directories/files to skip
- Add custom patterns:
```yaml
ignore_patterns:
  - node_modules
  - venv
  - my_custom_cache/  # Add custom
```

---

### 3. Output Configuration (`output`)

Controls what files are generated.

```yaml
output:
  format: json                          # Data format
  generate_html: true                   # Interactive report
  generate_markdown: true               # GitHub-friendly summary
  include_code_samples: true            # Example code snippets
  max_code_sample_lines: 50             # Lines per sample
  generate_executive_summary: true      # High-level overview
```

**Customize output:**
```yaml
output:
  generate_html: true          # Web report
  generate_markdown: true      # README-style report
  include_code_samples: false  # Skip code samples for privacy
  max_code_sample_lines: 100   # Longer samples
```

---

### 4. Portfolio Filtering (`blacklist`, `whitelist`)

Control which repositories appear in your portfolio.

#### Blacklist Mode (Default)
Exclude specific repositories:

```yaml
blacklist:
  - SiMLeng-Portfolio    # This meta-repo
  - test-repo           # Test projects
  - old-experiments     # Deprecated work
  - private-notes       # Personal docs

whitelist: []  # Empty = use blacklist mode
```

#### Whitelist Mode
Only include specified repositories:

```yaml
blacklist: []  # Ignored when whitelist is active

whitelist:
  - LLM-HypatiaX
  - DeepSeek-R1
  - deep-learning
  - tensorflow
```

**When to use each:**
- **Blacklist**: You want most repos shown, exclude a few
- **Whitelist**: You want only specific repos shown

---

### 5. Status Overrides (`status_overrides`)

Manually set project status instead of auto-detection.

```yaml
status_overrides:
  LLM-HypatiaX: current           # Active project
  old-ml-project: past            # Historical
  experimental-nlp: recent        # Recent work
```

**Status meanings:**
- **current**: Actively working on (< 30 days since update)
- **recent**: Recent work (30-180 days)
- **past**: Historical projects (> 180 days)

**Customize thresholds:**
```yaml
display:
  activity_thresholds:
    current: 60    # 60 days = current
    recent: 365    # 1 year = recent
```

---

### 6. Display Settings (`display`)

Configure how portfolio is presented.

```yaml
display:
  max_repos_per_category: 10     # Top N repos per category
  
  show_stars: true               # Show star count
  show_forks: true               # Show fork count
  show_language: true            # Primary language
  show_topics: true              # Repository topics
  show_last_updated: true        # Last update date
  
  show_code_samples: true        # Include code examples
  max_samples_per_repo: 3        # Samples per repository
  
  show_framework_badges: true    # Framework icons/badges
  show_activity_status: true     # Current/Recent/Past badges
```

---

### 7. Custom Categories (`custom_categories`)

Define how repositories are categorized.

```yaml
custom_categories:
  llm_and_ai:
    name: "AI & LLM Projects"
    description: "Large Language Models and generative AI"
    icon: "🧠"
    priority: 1                    # Display order
    required_frameworks:
      - transformers
      - langchain
    keywords:                      # Optional keyword matching
      - llm
      - gpt
```

**Add your own category:**
```yaml
custom_categories:
  my_category:
    name: "Web Development"
    description: "Full-stack web applications"
    icon: "🌐"
    priority: 8
    keywords:
      - react
      - django
      - fastapi
```

---

### 8. Highlighted Projects (`highlighted_projects`)

Pin specific projects to appear prominently.

```yaml
highlighted_projects:
  - LLM-HypatiaX         # Featured project 1
  - DeepSeek-R1          # Featured project 2
  - deep-learning        # Featured project 3
```

These appear at the top of your portfolio with special emphasis.

---

### 9. Scoring Weights (`scoring`)

Adjust how expertise score is calculated.

```yaml
scoring:
  stars_weight: 0.2              # 20% from stars
  forks_weight: 0.15             # 15% from forks
  commits_weight: 0.15           # 15% from commits
  contributors_weight: 0.1       # 10% from contributors
  code_quality_weight: 0.2       # 20% from code quality
  framework_diversity_weight: 0.1 # 10% from frameworks
  documentation_weight: 0.1      # 10% from docs
```

**Total must equal 1.0 (100%)**

**Example: Prioritize code quality over popularity:**
```yaml
scoring:
  stars_weight: 0.1              # Reduce star importance
  forks_weight: 0.1
  commits_weight: 0.15
  contributors_weight: 0.05
  code_quality_weight: 0.4       # Increase code quality
  framework_diversity_weight: 0.1
  documentation_weight: 0.1
```

---

### 10. Portfolio Metadata (`portfolio_metadata`)

Your contact information and portfolio description.

```yaml
portfolio_metadata:
  author: "Dr. Ruperto Pedro Bonet Chaple"
  email: "ruperto.bonet@modelphysmat.com"
  linkedin: "https://linkedin.com/in/your-profile"
  github: "https://github.com/sednabcn"
  
  title: "AI & Machine Learning Portfolio"
  subtitle: "Research in AI, LLMs, and Data Science"
  
  keywords:
    - Artificial Intelligence
    - Machine Learning
    - Deep Learning
```

---

## 🎨 Configuration Examples

### Example 1: Public Portfolio (Hide Private Work)

```yaml
blacklist:
  - private-client-work
  - confidential-research
  - personal-experiments

include_private: false
show_private_badge: false

highlighted_projects:
  - public-llm-project
  - open-source-contribution
```

### Example 2: Research Portfolio (Academic Focus)

```yaml
analysis:
  max_repos: 30
  min_stars: 5           # Only well-received work

scoring:
  stars_weight: 0.05     # Less emphasis on popularity
  code_quality_weight: 0.4   # More emphasis on quality
  documentation_weight: 0.2  # Strong documentation important

display:
  show_code_samples: true
  max_samples_per_repo: 5    # More code examples

custom_categories:
  research_papers:
    name: "Research Projects"
    description: "Academic research implementations"
    icon: "📚"
    priority: 1
```

### Example 3: Job Hunting Portfolio (Impress Recruiters)

```yaml
analysis:
  max_repos: 20          # Curated selection
  min_stars: 10          # Only popular projects

blacklist:
  - learning-tutorials
  - old-experiments
  - unfinished-projects

highlighted_projects:
  - flagship-project
  - award-winning-ml-model
  - production-system

display:
  show_stars: true
  show_forks: true
  show_activity_status: true
  max_repos_per_category: 5  # Top 5 only

sort_by: stars           # Most popular first
```

### Example 4: Comprehensive Archive (Everything)

```yaml
analysis:
  max_repos: 200         # Scan everything
  include_forks: true    # Include forks
  min_stars: 0           # All projects
  exclude_archived: false # Include archived

blacklist: []            # No exclusions
whitelist: []            # Include all

display:
  group_by_status: true  # Organize by activity
  show_last_updated: true
```

---

## 🔍 Debugging Configuration

### Check your configuration:

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test with small scan
python main.py --token YOUR_TOKEN --user sednabcn --max-repos 5
```

### Common Issues:

**1. Too many/few repositories**
- Adjust `analysis.max_repos`
- Check `blacklist`/`whitelist` settings
- Verify `min_stars` threshold

**2. Wrong project status**
- Use `status_overrides` for manual control
- Adjust `activity_thresholds`

**3. Missing frameworks**
- Add to appropriate `frameworks.*` list
- Check file extension in `code_extensions`

**4. Scan too slow**
- Reduce `max_repos`
- Increase `concurrent_requests` (careful with rate limits)
- Disable `include_code_samples`

---

## 📝 Best Practices

1. **Start Conservative**: Begin with `max_repos: 20` and increase
2. **Use Blacklist First**: Easier than whitelist for most cases
3. **Pin Your Best**: Use `highlighted_projects` for top 3-5 projects
4. **Customize Categories**: Align with your actual work areas
5. **Update Metadata**: Keep contact info current
6. **Test Locally**: Run local scan before GitHub Actions
7. **Version Control**: Commit config changes with descriptive messages

---

## 🆘 Quick Reference

| Task | Setting | Location |
|------|---------|----------|
| Exclude repo | Add to `blacklist` | Line 243 |
| Include only specific repos | Set `whitelist` | Line 256 |
| Scan more repos | Increase `max_repos` | Line 106 |
| Include forks | Set `include_forks: true` | Line 111 |
| Change project status | Add to `status_overrides` | Line 266 |
| Pin project | Add to `highlighted_projects` | Line 405 |
| Adjust scoring | Modify `scoring` weights | Line 189 |
| Add framework | Add to `frameworks.*` | Line 13-80 |

---

## 🚀 Ready to Configure!

1. Download `config_merged.yaml`
2. Rename to `config.yaml`
3. Customize settings (use examples above)
4. Test locally
5. Deploy to GitHub

Your portfolio, your rules! 🎉
