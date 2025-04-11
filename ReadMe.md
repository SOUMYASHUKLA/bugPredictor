
pip install gitpython

cd bugPredictor

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python
python3 --version

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3 data/count_commits.py
python3 data/generate_commit_data.py
python3 data/generate_bug_metadata.py


