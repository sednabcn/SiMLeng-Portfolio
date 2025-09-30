"""Simple repo analyzer using GitHub API"""
import aiohttp
import asyncio
from typing import Dict, List, Optional

class RepoAnalyzer:
    def __init__(self, github_token: str):
        self.github_token = github_token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    async def get_user_repositories(self, username: str, max_repos: int = 100, 
                                  include_forks: bool = False) -> List[Dict]:
        repos = []
        page = 1
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            while len(repos) < max_repos:
                url = f"{self.base_url}/users/{username}/repos"
                params = {"page": page, "per_page": 100, "sort": "updated"}
                
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        break
                    batch = await response.json()
                    if not batch:
                        break
                    
                    for repo in batch:
                        if include_forks or not repo.get('fork', False):
                            repos.append(repo)
                    page += 1
        
        return repos[:max_repos]
