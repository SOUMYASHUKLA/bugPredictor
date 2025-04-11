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

def extract_commit_data(repo_path: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    print(f"Opening repository at: {os.path.abspath(repo_path)}")
    repo = git.Repo(repo_path)
    
    # Get default branch
    branch = get_default_branch(repo)
    print(f"Using branch: {branch}")
    
    # Convert dates to timezone-aware datetime objects
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    start_dt = pytz.UTC.localize(start_dt)
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    end_dt = pytz.UTC.localize(end_dt)
    
    print(f"Extracting commits from {start_date} to {end_date}")
    
    commits_data = []
    try:
        for commit in repo.iter_commits(branch):
            commit_date = commit.committed_datetime
            
            # Skip commits outside our date range
            if commit_date < start_dt or commit_date > end_dt:
                continue
                
            try:
                # Get the diff for this commit
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
                
                if len(commits_data) % 10 == 0:  # More frequent updates since we expect fewer commits
                    print(f"Processed {len(commits_data)} commits from 2024-2025...")
                    
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
    
    # Navigate up two directories to find tomcat
    repo_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'tomcat'))
    
    if not os.path.exists(repo_path):
        print(f"Error: Could not find Tomcat repository at {repo_path}")
        return
        
    print(f"Found Tomcat repository at: {repo_path}")
    
    # Extract commits for 2024-2025 only
    commits_data = extract_commit_data(
        repo_path=repo_path,
        start_date='2024-01-01',
        end_date='2025-01-01'
    )
    
    if not commits_data:
        print("No commits were extracted. Please check the repository path and permissions.")
        return
    
    # Save to JSON file
    output_file = data_dir / 'commits_data_2024_2025.json'
    with open(output_file, 'w') as f:
        json.dump(commits_data, f, indent=2)
    
    print(f"\nExtracted {len(commits_data)} commits from 2024-2025")
    print(f"Data saved to {output_file}")
    
    # Print some statistics
    if commits_data:
        total_files_changed = sum(c['stats']['total_files_changed'] for c in commits_data)
        total_lines_added = sum(c['stats']['total_lines_added'] for c in commits_data)
        total_lines_removed = sum(c['stats']['total_lines_removed'] for c in commits_data)
        
        print("\nCommit Statistics:")
        print(f"Total files changed: {total_files_changed}")
        print(f"Total lines added: {total_lines_added}")
        print(f"Total lines removed: {total_lines_removed}")
        
        # Show date range of collected commits
        dates = [datetime.fromisoformat(c['date']) for c in commits_data]
        if dates:
            print(f"\nDate range of commits:")
            print(f"First commit: {min(dates).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Last commit: {max(dates).strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()