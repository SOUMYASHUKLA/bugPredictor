
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


python llm/train_bug_predictor.py

python llm/predict_bugs.py --model-path checkpoints/model_epoch_5.pt --commit-message "Your commit message" --code-changes "Your code changes"

## use generated model for prediction

# 1. Interactive Mode (easiest for testing):
python llm/predict.py --interactive --model-path llm/checkpoints/model_epoch_5.pt

# 2. Using JSON File:
# Create a JSON file with commit data
echo '{
    "message": "Fix null pointer exception in UserService",
    "code_changes": "public void processUser(User user) {\n  if (user != null) {\n    user.process();\n  }\n}"
}' > commit.json

# Make prediction
python llm/predict.py --json-input commit.json --model-path llm/checkpoints/model_epoch_5.pt

# 3. Direct JSON Input:
python llm/predict.py --commit-data '{"message": "Fix bug in login", "code_changes": "if (user.isAuthenticated()) {...}"}' --model-path llm/checkpoints/model_epoch_5.pt
