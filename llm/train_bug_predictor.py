import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CommitDataset(Dataset):
    def __init__(self, commits_data: List[Dict], bug_metadata: List[Dict], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create bug commit mapping
        bug_commits = set()
        for bug in bug_metadata:
            for commit in bug['related_commits']:
                bug_commits.add(commit['hash'])
        
        # Prepare data
        self.examples = []
        for commit in commits_data:
            # Combine commit message and code changes
            code_changes = "\n".join([
                f"File: {change['file_path']}\n{change['new_content']}"
                for change in commit['changes'][:3]  # Limit to first 3 files for memory
            ])
            
            text = f"Commit Message: {commit['message']}\nCode Changes: {code_changes}"
            
            # Label is 1 if commit is associated with a bug
            label = 1.0 if commit['hash'] in bug_commits else 0.0
            
            self.examples.append((text, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BugPredictor(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        super().__init__()
        self.code_encoder = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)  # 768 is CodeBERT's hidden size
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.code_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.classifier(x)
        return self.sigmoid(x)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    checkpoint_dir: str
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({'training_loss': f'{train_loss/train_steps:.3f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs.squeeze(), labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                # Calculate accuracy
                predictions = (outputs.squeeze() > 0.5).float()
                correct_preds += (predictions == labels).sum().item()
                total_preds += labels.size(0)
        
        avg_val_loss = val_loss / val_steps
        accuracy = correct_preds / total_preds
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {train_loss/train_steps:.3f}')
        print(f'Average validation loss: {avg_val_loss:.3f}')
        print(f'Validation accuracy: {accuracy:.3f}')
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

def main():
    # Create necessary directories
    data_dir = Path('..', 'data')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    with open(data_dir / 'commits_data_2020_2024.json', 'r') as f:
        commits_data = json.load(f)
    with open(data_dir / 'bug_metadata_2020_2024.json', 'r') as f:
        bug_metadata = json.load(f)
    
    print(f"Loaded {len(commits_data)} commits and {len(bug_metadata)} bugs")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # Create dataset
    dataset = CommitDataset(commits_data, bug_metadata, tokenizer)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BugPredictor()
    model.to(device)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=5,
        learning_rate=2e-5,
        device=device,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()