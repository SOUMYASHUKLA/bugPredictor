import json
import os
from pprint import pprint

def show_sampleData():
    try:
        with open('data/commits_data.json', 'r') as f:
            data = json.load(f)
            if data:
                print("\nTotal commits in file:", len(data))
                print("\nSample commit data (first commit):")
                print("=" * 80)
                pprint(data[0], indent=2, width=120)
                print("=" * 80)
            else:
                print("No commits found in file")
    except FileNotFoundError:
        print("Error: commits_data.json not found in data directory")
    except json.JSONDecodeError:
        print("Error: Invalid JSON file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    show_sampleData() 