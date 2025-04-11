import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

def extract_bug_references(commit_message: str) -> List[str]:
    """Extract bug IDs from commit messages using common patterns."""
    patterns = [
        r'(?i)fixes?\s+#?(\d+)',  # matches: fix #123, fixes 123
        r'(?i)closes?\s+#?(\d+)',  # matches: close #123, closes 123
        r'(?i)resolves?\s+#?(\d+)',  # matches: resolve #123, resolves 123
        r'(?i)bug\s+#?(\d+)',  # matches: bug #123, bug 123
        r'(?i)issue\s+#?(\d+)',  # matches: issue #123, issue 123
        r'TOMCAT-(\d+)'  # matches TOMCAT-123
    ]
    
    bug_ids = set()
    for pattern in patterns:
        matches = re.finditer(pattern, commit_message)
        bug_ids.update(match.group(1) for match in matches)
    
    return list(bug_ids)

def analyze_commits_for_bugs(commits_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze commits to extract bug metadata."""
    bugs_data = {}
    
    for commit in commits_data:
        bug_refs = extract_bug_references(commit['message'])
        
        for bug_id in bug_refs:
            if bug_id not in bugs_data:
                bugs_data[bug_id] = {
                    'bug_id': bug_id,
                    'first_mention_date': commit['date'],
                    'related_commits': [],
                    'affected_files': set(),
                    'total_changes': {
                        'files_changed': 0,
                        'lines_added': 0,
                        'lines_removed': 0
                    }
                }
            
            # Add commit reference
            bugs_data[bug_id]['related_commits'].append(commit['hash'])
            
            # Update statistics
            bugs_data[bug_id]['total_changes']['files_changed'] += commit['stats']['total_files_changed']
            bugs_data[bug_id]['total_changes']['lines_added'] += commit['stats']['total_lines_added']
            bugs_data[bug_id]['total_changes']['lines_removed'] += commit['stats']['total_lines_removed']
            
            # Add affected files
            for change in commit['changes']:
                bugs_data[bug_id]['affected_files'].add(change['file_path'])
    
    # Convert sets to lists for JSON serialization
    for bug_data in bugs_data.values():
        bug_data['affected_files'] = list(bug_data['affected_files'])
    
    return list(bugs_data.values())

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Load commits data
    commits_file = data_dir / 'commits_data.json'
    with open(commits_file, 'r') as f:
        commits_data = json.load(f)
    
    # Extract bug metadata
    bugs_data = analyze_commits_for_bugs(commits_data)
    
    # Save to JSON file
    output_file = data_dir / 'bug_metadata.json'
    with open(output_file, 'w') as f:
        json.dump(bugs_data, f, indent=2)
    
    print(f"\nExtracted metadata for {len(bugs_data)} bugs")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main() 