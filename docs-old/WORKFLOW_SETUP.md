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
