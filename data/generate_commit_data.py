import git
import json
from datetime import datetime
import os
from typing import Dict, List, Any
from pathlib import Path
import pytz

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

def get_file_changes(diff_index) -> List[Dict[str, Any]]:
    changes = []
    for diff in diff_index:
        try:
            # For binary files or deletions/new files, this might fail
            a_blob = diff.a_blob
            b_blob = diff.b_blob
            
            if a_blob and b_blob:  # Modified file
                old_content = a_blob.data_stream.read().decode('utf-8', errors='ignore')
                new_content = b_blob.data_stream.read().decode('utf-8', errors='ignore')
            elif b_blob:  # New file
                new_content = b_blob.data_stream.read().decode('utf-8', errors='ignore')
                old_content = ""
            elif a_blob:  # Deleted file
                old_content = a_blob.data_stream.read().decode('utf-8', errors='ignore')
                new_content = ""
            else:
                continue

            changes.append({
                'file_path': diff.b_path or diff.a_path,
                'change_type': diff.change_type,
                'old_content': old_content,
                'new_content': new_content,
                'lines_added': len([l for l in new_content.splitlines() if l.strip()]),
                'lines_removed': len([l for l in old_content.splitlines() if l.strip()])
            })
        except Exception as e:
            print(f"Error processing file {diff.b_path or diff.a_path}: {str(e)}")
            continue
            
    return changes

def extract_commit_data(repo_path: str, until_date: str) -> List[Dict[str, Any]]:
    print(f"Attempting to open repository at: {os.path.abspath(repo_path)}")
    try:
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Error: {repo_path} is not a valid Git repository")
        print("Current working directory:", os.getcwd())
        print("Available files/directories:")
        print(os.listdir('.'))
        return []
    
    commits_data = []
    
    # Get the default branch
    default_branch = get_default_branch(repo)
    print(f"Using branch: {default_branch}")
    
    # Convert until_date string to timezone-aware datetime object
    until_dt = datetime.strptime(until_date, '%Y-%m-%d')
    until_dt = pytz.UTC.localize(until_dt)
    
    print("Starting commit extraction...")
    print(f"Extracting commits until: {until_date}")
    
    try:
        for commit in repo.iter_commits(default_branch):
            if commit.committed_datetime > until_dt:
                continue
                
            try:
                if len(commit.parents) > 0:
                    diff_index = commit.parents[0].diff(commit)
                else:
                    diff_index = commit.diff(git.NULL_TREE)
                    
                changes = get_file_changes(diff_index)
                
                commit_data = {
                    'hash': commit.hexsha,
                    'author': {
                        'name': commit.author.name,
                        'email': commit.author.email
                    },
                    'date': commit.committed_datetime.isoformat(),
                    'message': commit.message,
                    'changes': changes,
                    'stats': {
                        'total_files_changed': len(changes),
                        'total_lines_added': sum(c['lines_added'] for c in changes),
                        'total_lines_removed': sum(c['lines_removed'] for c in changes)
                    }
                }
                
                commits_data.append(commit_data)
                
                if len(commits_data) % 100 == 0:
                    print(f"Processed {len(commits_data)} commits...")
                    
            except Exception as e:
                print(f"Error processing commit {commit.hexsha}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error iterating commits: {str(e)}")
        return []
    
    return commits_data

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Print current working directory and available paths
    print("Current working directory:", os.getcwd())
    print("Available directories:", os.listdir('..'))
    
    ##TODO: get remote repo from gitHub
    repo_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'tomcat'))
    
    if not os.path.exists(repo_path):
        print(f"Error: Could not find Tomcat repository at {repo_path}")
        return
        
    print(f"Found Tomcat repository at: {repo_path}")
    
    ##TODO: Extract commits in date range
    commits_data = extract_commit_data(
        repo_path=repo_path,
        until_date='2025-01-01'
    )
    
    if not commits_data:
        print("No commits were extracted. Please check the repository path and permissions.")
        return
    
    # Save to JSON file
    output_file = data_dir / 'commits_data.json'
    with open(output_file, 'w') as f:
        json.dump(commits_data, f, indent=2)
    
    print(f"\nExtracted {len(commits_data)} commits")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main() 