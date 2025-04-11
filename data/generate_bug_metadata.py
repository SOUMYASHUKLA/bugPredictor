import json
import re
from datetime import datetime
import os
from typing import Dict, List, Any
from pathlib import Path

def extract_bug_references(commit_message: str) -> List[str]:
    """Extract bug IDs from commit messages using common patterns."""
    patterns = [
        r'(?i)fixes?\s+#?(\d+)',  # matches: fix #123, fixes 123
        r'(?i)closes?\s+#?(\d+)',  # matches: close #123, closes 123
        r'(?i)resolves?\s+#?(\d+)',  # matches: resolve #123, resolves 123
        r'(?i)bug\s+#?(\d+)',  # matches: bug #123, bug 123
        r'(?i)issue\s+#?(\d+)',  # matches: issue #123, issue 123
        r'TOMCAT-(\d+)',  # matches TOMCAT-123
        r'(?i)CVE-(\d{4}-\d+)'  # matches CVE-2024-1234
    ]
    
    bug_ids = set()
    for pattern in patterns:
        matches = re.finditer(pattern, commit_message)
        bug_ids.update(match.group(1) for match in matches)
    
    return list(bug_ids)

def analyze_bug_patterns(commit: Dict) -> Dict[str, Any]:
    """Analyze commit for common bug patterns and security issues."""
    patterns = {
        'null_pointer': r'NullPointerException|null\s+pointer',
        'memory_leak': r'memory\s+leak|OutOfMemoryError',
        'race_condition': r'race\s+condition|concurrent|synchronization\s+issue',
        'security_vulnerability': r'security|vulnerability|CVE|exploit',
        'performance_issue': r'performance|slow|timeout|deadlock',
        'resource_leak': r'resource\s+leak|connection\s+leak|file\s+handle',
    }
    
    found_patterns = {}
    combined_text = f"{commit['message']} {' '.join(str(c.get('new_content', '')) for c in commit['changes'])}"
    
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            found_patterns[pattern_name] = True
    
    return found_patterns

def analyze_commits_for_bugs(commits_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze commits to extract bug metadata."""
    bugs_data = {}
    
    print("Analyzing commits for bug patterns...")
    
    for commit in commits_data:
        # Extract bug references from commit message
        bug_refs = extract_bug_references(commit['message'])
        
        # Analyze for bug patterns
        bug_patterns = analyze_bug_patterns(commit)
        
        # If we found bug references or patterns, create/update bug entry
        is_bug_commit = bool(bug_refs) or bool(bug_patterns)
        
        if is_bug_commit:
            # Create unique bug ID if no specific reference found
            if not bug_refs:
                bug_refs = [f"AUTO-{commit['hash'][:8]}"]
            
            for bug_id in bug_refs:
                if bug_id not in bugs_data:
                    bugs_data[bug_id] = {
                        'bug_id': bug_id,
                        'first_seen_date': commit['date'],
                        'related_commits': [],
                        'affected_files': set(),
                        'bug_patterns': set(),
                        'total_changes': {
                            'files_changed': 0,
                            'lines_added': 0,
                            'lines_removed': 0
                        }
                    }
                
                bug_data = bugs_data[bug_id]
                
                # Add commit reference
                bug_data['related_commits'].append({
                    'hash': commit['hash'],
                    'date': commit['date'],
                    'message': commit['message'],
                    'author': commit['author']
                })
                
                # Update statistics
                bug_data['total_changes']['files_changed'] += commit['stats']['total_files_changed']
                bug_data['total_changes']['lines_added'] += commit['stats']['total_lines_added']
                bug_data['total_changes']['lines_removed'] += commit['stats']['total_lines_removed']
                
                # Add affected files
                for change in commit['changes']:
                    bug_data['affected_files'].add(change['file_path'])
                
                # Add bug patterns
                bug_data['bug_patterns'].update(bug_patterns.keys())
    
    # Convert sets to lists for JSON serialization
    for bug_data in bugs_data.values():
        bug_data['affected_files'] = list(bug_data['affected_files'])
        bug_data['bug_patterns'] = list(bug_data['bug_patterns'])
    
    return list(bugs_data.values())

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Load commits data
    commits_file = data_dir / 'commits_data_2024_2025.json'
    
    if not commits_file.exists():
        print(f"Error: Commits data file not found at {commits_file}")
        print("Please run generate_commit_data.py first")
        return
    
    print(f"Loading commits data from {commits_file}")
    with open(commits_file, 'r') as f:
        commits_data = json.load(f)
    
    print(f"Loaded {len(commits_data)} commits")
    
    # Extract bug metadata
    bugs_data = analyze_commits_for_bugs(commits_data)
    
    # Save to JSON file
    output_file = data_dir / 'bug_metadata_2024_2025.json'
    with open(output_file, 'w') as f:
        json.dump(bugs_data, f, indent=2)
    
    # Print statistics
    print(f"\nExtracted metadata for {len(bugs_data)} bugs")
    print(f"Data saved to {output_file}")
    
    if bugs_data:
        # Calculate some statistics
        total_commits = sum(len(bug['related_commits']) for bug in bugs_data)
        total_files = sum(len(bug['affected_files']) for bug in bugs_data)
        
        print("\nBug Statistics:")
        print(f"Total bugs found: {len(bugs_data)}")
        print(f"Total bug-related commits: {total_commits}")
        print(f"Total affected files: {total_files}")
        
        # Bug pattern distribution
        pattern_counts = {}
        for bug in bugs_data:
            for pattern in bug['bug_patterns']:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if pattern_counts:
            print("\nBug Pattern Distribution:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"{pattern}: {count} bugs")

if __name__ == "__main__":
    main() 