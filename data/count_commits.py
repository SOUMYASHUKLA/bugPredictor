import git
from datetime import datetime
import pytz
import os

def get_default_branch(repo) -> str:
    """Get the default branch name (main or master)"""
    try:
        # Try main first
        repo.refs['main']
        return 'main'
    except KeyError:
        try:
            # Try master
            repo.refs['master']
            return 'master'
        except KeyError:
            # If neither exists, get the current branch
            return repo.active_branch.name

def count_commits(repo_path: str, until_date: str) -> int:
    print(f"Opening repository at: {os.path.abspath(repo_path)}")
    repo = git.Repo(repo_path)
    
    # Get default branch
    branch = get_default_branch(repo)
    print(f"Using branch: {branch}")
    
    # Convert until_date string to timezone-aware datetime object
    until_dt = datetime.strptime(until_date, '%Y-%m-%d')
    until_dt = pytz.UTC.localize(until_dt)
    
    print(f"Counting commits until {until_date}...")
    
    count = 0
    for commit in repo.iter_commits(branch):
        if commit.committed_datetime > until_dt:
            continue
        count += 1
        if count % 1000 == 0:
            print(f"Counted {count} commits so far...")
    
    return count

def main():
    # Navigate up two directories to find tomcat
    repo_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'tomcat'))
    
    if not os.path.exists(repo_path):
        print(f"Error: Could not find Tomcat repository at {repo_path}")
        print("Current working directory:", os.getcwd())
        return
    
    print(f"Found Tomcat repository at: {repo_path}")
    until_date = '2025-01-01'
    
    try:
        total_commits = count_commits(repo_path, until_date)
        print(f"\nTotal number of commits until {until_date}: {total_commits}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Current working directory:", os.getcwd())
        print("Available directories:", os.listdir('..'))

if __name__ == "__main__":
    main() 