#!/usr/bin/env python3
"""
AI Portfolio Scanner - Main Entry Point
Analyzes GitHub repositories for AI/ML expertise
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import ConfigLoader
from repo_analyzer import RepoAnalyzer
from portfolio_builder import PortfolioBuilder
from data_models import ScanConfig


def setup_logging(level: str = "INFO", save_logs: bool = True):
    """Configure logging for the scanner"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Console handler
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler()]
    )
    
    # File handler
    if save_logs:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"portfolio_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
        logging.info(f"Logging to {log_file}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scan GitHub repositories for AI/ML expertise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a specific user
  python main.py --token YOUR_TOKEN --user octocat
  
  # Scan specific repositories
  python main.py --token YOUR_TOKEN --repos "owner/repo1,owner/repo2"
  
  # Use custom config
  python main.py --token YOUR_TOKEN --user octocat --config custom_config.yaml
        """
    )
    
    # Authentication
    parser.add_argument(
        "--token",
        required=True,
        help="GitHub Personal Access Token"
    )
    
    # Target selection (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--user",
        help="GitHub username to scan"
    )
    target_group.add_argument(
        "--repos",
        help="Comma-separated repository URLs (owner/repo format)"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="portfolio-output",
        help="Output directory for results (default: portfolio-output)"
    )
    
    # Optional overrides
    parser.add_argument(
        "--max-repos",
        type=int,
        help="Maximum repositories to scan (overrides config)"
    )
    
    parser.add_argument(
        "--min-stars",
        type=int,
        help="Minimum stars filter (overrides config)"
    )
    
    parser.add_argument(
        "--include-forks",
        action="store_true",
        help="Include forked repositories (overrides config)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Load configuration
    try:
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Setup logging
    log_config = config.get("logging", {})
    setup_logging(
        level=args.log_level or log_config.get("level", "INFO"),
        save_logs=log_config.get("save_logs", True)
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("AI Portfolio Scanner Started")
    logger.info("=" * 70)
    
    # Apply CLI overrides
    analysis_config = config.get("analysis", {})
    if args.max_repos:
        analysis_config["max_repos"] = args.max_repos
    if args.min_stars is not None:
        analysis_config["min_stars"] = args.min_stars
    if args.include_forks:
        analysis_config["include_forks"] = True
    
    # Create scan configuration
    scan_config = ScanConfig(
        github_token=args.token,
        target_user=args.user,
        target_repos=args.repos.split(",") if args.repos else None,
        frameworks=config.get("frameworks", {}),
        analysis_config=analysis_config,
        output_config=config.get("output", {}),
        output_dir=Path(args.output_dir)
    )
    
    try:
        # Initialize analyzer
        logger.info(f"Target: {args.user or args.repos}")
        logger.info(f"Max repos: {scan_config.analysis_config.get('max_repos', 'unlimited')}")
        logger.info(f"Output directory: {scan_config.output_dir}")
        
        analyzer = RepoAnalyzer(scan_config)
        
        # Scan repositories
        logger.info("\n🔍 Starting repository scan...")
        scan_results = analyzer.scan()
        
        if not scan_results:
            logger.warning("⚠️  No repositories found or scanned")
            return
        
        logger.info(f"✅ Scanned {len(scan_results)} repositories")
        
        # Build portfolio
        logger.info("\n📊 Building portfolio...")
        portfolio_builder = PortfolioBuilder(scan_config, scan_results)
        portfolio_data = portfolio_builder.build()
        
        # Generate outputs
        logger.info("\n💾 Generating output files...")
        output_files = portfolio_builder.generate_outputs(portfolio_data)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Portfolio scan completed successfully!")
        logger.info("=" * 70)
        logger.info("\n📁 Generated files:")
        for file_path in output_files:
            logger.info(f"  • {file_path}")
        
        # Print summary statistics
        stats = portfolio_data.get("statistics", {})
        logger.info("\n📈 Summary:")
        logger.info(f"  • Repositories scanned: {stats.get('repositories_scanned', 0)}")
        logger.info(f"  • AI/ML relevant: {stats.get('relevant_repositories', 0)}")
        logger.info(f"  • Expertise score: {stats.get('overall_expertise_score', 0):.1f}/10")
        logger.info(f"  • Total files analyzed: {stats.get('total_files_analyzed', 0)}")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Scan failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
