import json
import os

def view_commits_sample(file_path, num_commits=5):
    """View a sample of commits from the JSON file"""
    print(f"Reading from: {os.path.abspath(file_path)}")
    
    try:
        with open(file_path, 'r') as f:
            # Read first 1000 characters to find the first few complete commits
            data = json.load(f)
            
        print(f"\nTotal number of commits in file: {len(data)}")
        print("\nShowing first", num_commits, "commits:\n")
        
        for i, commit in enumerate(data[:num_commits]):
            print(f"Commit {i+1}:")
            print(f"Hash: {commit['hash']}")
            print(f"Author: {commit['author']['name']} <{commit['author']['email']}>")
            print(f"Date: {commit['date']}")
            print("Message:")
            print(commit['message'].strip())
            print(f"Files changed: {commit['stats']['total_files_changed']}")
            print(f"Lines added: {commit['stats']['total_lines_added']}")
            print(f"Lines removed: {commit['stats']['total_lines_removed']}")
            print("-" * 80 + "\n")
            
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    file_path = os.path.join('data', 'commits_data.json')
    view_commits_sample(file_path)

if __name__ == "__main__":
    main() 