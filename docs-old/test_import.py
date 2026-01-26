mport sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

try:
    from repo_analyzer import RepoAnalyzer
    print("✓ repo_analyzer imported")
    from code_analyzer import CodeAnalyzer
    print("✓ code_analyzer imported")
    from framework_detector import FrameworkDetector
    print("✓ framework_detector imported")
    from portfolio_builder import PortfolioBuilder
    print("✓ portfolio_builder imported")
    from data_models import RepositoryData
    print("✓ data_models imported")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
