"""Configuration loader for portfolio settings"""
import yaml
import os
from pathlib import Path
from typing import Dict, List, Set

class PortfolioConfig:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent / '.github' / 'portfolio-config.yml'
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"Warning: Config file not found at {self.config_path}, using defaults")
            return self._default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            return {**self._default_config(), **config}
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'blacklist': [],
            'whitelist': [],
            'status_overrides': {},
            'min_relevance_score': 1.0,
            'include_forks': False,
            'include_private': False,
            'show_private_badge': True,
            'group_by_status': True,
            'sort_by': 'score'
        }
    
    def get_blacklist(self) -> Set[str]:
        """Get set of blacklisted repository names"""
        # Combine config file blacklist with environment variable
        blacklist = set(self.config.get('blacklist', []))
        
        # Add from environment variable
        env_blacklist = os.getenv('REPO_BLACKLIST', '')
        if env_blacklist:
            blacklist.update(name.strip() for name in env_blacklist.split(',') if name.strip())
        
        return blacklist
    
    def get_whitelist(self) -> Set[str]:
        """Get set of whitelisted repository names (if using whitelist mode)"""
        return set(self.config.get('whitelist', []))
    
    def is_whitelisted(self, repo_name: str) -> bool:
        """Check if repo is whitelisted (if whitelist mode is active)"""
        whitelist = self.get_whitelist()
        if not whitelist:
            return True  # Whitelist mode not active
        return repo_name in whitelist
    
    def is_blacklisted(self, repo_name: str) -> bool:
        """Check if repo is blacklisted"""
        return repo_name in self.get_blacklist()
    
    def should_include_repo(self, repo_name: str) -> bool:
        """Determine if repo should be included based on whitelist/blacklist"""
        if self.is_blacklisted(repo_name):
            return False
        return self.is_whitelisted(repo_name)
    
    def get_status_override(self, repo_name: str) -> str:
        """Get manual status override for a repo, if any"""
        return self.config.get('status_overrides', {}).get(repo_name)
    
    def get_min_score(self) -> float:
        """Get minimum relevance score threshold"""
        return self.config.get('min_relevance_score', 1.0)
    
    def include_forks(self) -> bool:
        """Check if forks should be included"""
        return self.config.get('include_forks', False)
    
    def include_private(self) -> bool:
        """Check if private repos should be included"""
        # Environment variable takes precedence
        env_val = os.getenv('INCLUDE_PRIVATE_REPOS', '').lower()
        if env_val in ('true', 'false'):
            return env_val == 'true'
        return self.config.get('include_private', False)
    
    def get_sort_by(self) -> str:
        """Get sorting preference"""
        return self.config.get('sort_by', 'score')
