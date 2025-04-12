import torch
from train_bug_predictor import BugPredictor
from transformers import RobertaTokenizer
import argparse

def predict_bug_probability(
    model_path: str,
    commit_message: str,
    code_changes: str,
    device: torch.device = None
) -> float:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BugPredictor()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Prepare input
    text = f"Commit Message: {commit_message}\nCode Changes: {code_changes}"
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Make prediction
    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask)
        probability = outputs.squeeze().item()
    
    return probability

def main():
    parser = argparse.ArgumentParser(description='Predict bug probability for a commit')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--commit-message', type=str, required=True, help='Commit message')
    parser.add_argument('--code-file', type=str, help='File containing code changes')
    parser.add_argument('--code-changes', type=str, help='Direct code changes')
    
    args = parser.parse_args()
    
    # Get code changes from file or direct input
    if args.code_file:
        with open(args.code_file, 'r') as f:
            code_changes = f.read()
    else:
        code_changes = args.code_changes or ''
    
    # Make prediction
    probability = predict_bug_probability(
        model_path=args.model_path,
        commit_message=args.commit_message,
        code_changes=code_changes
    )
    
    print(f"\nBug Probability: {probability:.2%}")
    
    # Provide interpretation
    if probability > 0.8:
        print("HIGH RISK: This commit has a high probability of containing bugs!")
    elif probability > 0.5:
        print("MEDIUM RISK: This commit might contain bugs.")
    else:
        print("LOW RISK: This commit appears to be relatively safe.")

if __name__ == "__main__":
    main()