import torch
from train_bug_predictor import BugPredictor
from transformers import RobertaTokenizer
import argparse
import json
from pathlib import Path

class BugPredictionService:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print("Loading model...")
        self.model = BugPredictor()
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    def predict(self, commit_message: str, code_changes: str) -> dict:
        # Prepare input
        text = f"Commit Message: {commit_message}\nCode Changes: {code_changes}"
        encoding = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            outputs = self.model(input_ids, attention_mask)
            probability = outputs.squeeze().item()
        
        # Determine risk level and provide explanation
        if probability > 0.8:
            risk_level = "HIGH"
            explanation = "This commit has a high probability of containing bugs!"
        elif probability > 0.5:
            risk_level = "MEDIUM"
            explanation = "This commit might contain bugs."
        else:
            risk_level = "LOW"
            explanation = "This commit appears to be relatively safe."
        
        return {
            "probability": probability,
            "risk_level": risk_level,
            "explanation": explanation
        }

def format_code_changes(changes: list) -> str:
    """Format code changes into a readable string"""
    formatted = []
    for change in changes:
        formatted.append(f"File: {change['file']}")
        formatted.append("Changes:")
        formatted.append(change['content'])
        formatted.append("-" * 40)
    return "\n".join(formatted)

def main():
    parser = argparse.ArgumentParser(description='Predict bugs in commits')
    parser.add_argument('--model-path', type=str, 
                       default='checkpoints/model_epoch_5.pt',
                       help='Path to trained model checkpoint')
    
    # Input methods
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--interactive', action='store_true',
                      help='Run in interactive mode')
    group.add_argument('--json-input', type=str,
                      help='JSON file containing commit data')
    group.add_argument('--commit-data', type=str,
                      help='Direct JSON string of commit data')
    
    args = parser.parse_args()
    
    # Initialize prediction service
    predictor = BugPredictionService(args.model_path)
    
    if args.interactive:
        print("\n=== Bug Predictor Interactive Mode ===")
        while True:
            print("\nEnter commit details (Ctrl+C to exit):")
            try:
                commit_message = input("\nCommit message: ").strip()
                print("\nEnter code changes (empty line to finish):")
                code_changes = []
                while True:
                    line = input()
                    if not line:
                        break
                    code_changes.append(line)
                
                result = predictor.predict(
                    commit_message=commit_message,
                    code_changes="\n".join(code_changes)
                )
                
                print("\nPrediction Results:")
                print(f"Risk Level: {result['risk_level']}")
                print(f"Bug Probability: {result['probability']:.2%}")
                print(f"Explanation: {result['explanation']}")
                
                print("\nAnalyze another commit? (y/n)")
                if input().lower() != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    else:
        # Process JSON input
        if args.json_input:
            with open(args.json_input, 'r') as f:
                commit_data = json.load(f)
        else:
            commit_data = json.loads(args.commit_data)
        
        # Make prediction
        result = predictor.predict(
            commit_message=commit_data['message'],
            code_changes=commit_data.get('code_changes', '')
        )
        
        # Print results
        print("\nPrediction Results:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main() 