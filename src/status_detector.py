"""Project status detection for portfolio."""
from datetime import datetime, timedelta
from typing import Optional
from data_models import RepositoryData, PortfolioConfig

def determine_project_status(
    repo: RepositoryData, 
    config: Optional[PortfolioConfig] = None
) -> str:
    """
    Determine project status based on last update date and configuration.
    
    Status definitions:
    - CURRENT: Updated in last 30 days OR has 'current-project' topic
    - RECENT: Updated in last 6 months
    - PAST: Not updated in 6+ months OR has 'past-project'/'completed' topic
    
    Returns:
        str: 'current', 'recent', or 'past'
    """
    # Check for manual override in config
    if config:
        override = config.get_status_override(repo.name)
        if override:
            return override.lower()
    
    # Check for topic-based status
    topics_lower = [t.lower() for t in repo.topics]
    
    if 'current-project' in topics_lower:
        return 'current'
    
    if 'past-project' in topics_lower or 'completed' in topics_lower or 'archived' in topics_lower:
        return 'past'
    
    # Determine by last update date
    try:
        # Use pushed_at if available, otherwise updated_at
        date_str = repo.pushed_at or repo.updated_at
        last_update = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
        now = datetime.utcnow()
        days_since_update = (now - last_update).days
        
        if days_since_update <= 30:
            return 'current'
        elif days_since_update <= 180:  # 6 months
            return 'recent'
        else:
            return 'past'
    except (ValueError, TypeError):
        # If date parsing fails, default to 'recent'
        return 'recent'


def get_status_badge(status: str) -> dict:
    """
    Get display information for a status.
    
    Returns:
        dict: {'emoji': str, 'label': str, 'color': str}
    """
    status_info = {
        'current': {
            'emoji': '🟢',
            'label': 'CURRENT',
            'color': '#22c55e',
            'description': 'Active development'
        },
        'recent': {
            'emoji': '🟡',
            'label': 'RECENT',
            'color': '#eab308',
            'description': 'Recently updated'
        },
        'past': {
            'emoji': '⚪',
            'label': 'PAST',
            'color': '#94a3b8',
            'description': 'Completed or archived'
        }
    }
    
    return status_info.get(status, status_info['recent'])


def group_by_status(repos: list[RepositoryData]) -> dict:
    """
    Group repositories by their status.
    
    Returns:
        dict: {'current': [...], 'recent': [...], 'past': [...]}
    """
    groups = {
        'current': [],
        'recent': [],
        'past': []
    }
    
    for repo in repos:
        status = repo.project_status or 'recent'
        groups[status].append(repo)
    
    return groups


def get_status_summary(repos: list[RepositoryData]) -> dict:
    """
    Get summary statistics for project statuses.
    
    Returns:
        dict: Summary with counts and percentages
    """
    total = len(repos)
    if total == 0:
        return {
            'total': 0,
            'current': 0,
            'recent': 0,
            'past': 0
        }
    
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


def sort_by_status(repos: list[RepositoryData]) -> list[RepositoryData]:
    """
    Sort repositories with current first, then recent, then past.
    Within each group, sort by relevance score.
    """
    status_order = {'current': 0, 'recent': 1, 'past': 2}
    
    return sorted(
        repos,
        key=lambda r: (
            status_order.get(r.project_status or 'recent', 1),
            -r.relevance_score,
            -r.stargazers_count
        )
    )
