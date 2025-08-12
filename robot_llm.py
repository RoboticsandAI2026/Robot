import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import os

class RobotTokenizer:
    def __init__(self):
        # Enhanced vocabulary with more explicit tokens
        self.vocab = {
            # Special tokens
            '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3,
            # Action verbs with responses
            'Move': 4, 'Go': 5, 'Clean': 6, 'Moving': 7, 'Cleaning': 8,
            # Important locations first (Home before rooms)
            'Home': 9, 'Room': 10,
            # Location specifiers
            'a': 11, 'b': 12, 'c': 13, 'd': 14,
            # Connectors and articles
            'To': 15, 'The': 16, 'And': 17
        }
        self.id2word = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text.split()]

    def decode(self, tokens):
        words = []
        for token in tokens:
            if token in [self.vocab['<PAD>'], self.vocab['<START>']]:
                continue
            if token == self.vocab['<END>']:
                break
            word = self.id2word.get(token, '<UNK>')
            if word.startswith('<') and word.endswith('>'):
                continue
            words.append(word)
        if not words:
            return "No response generated."
        return ' '.join(words)

def create_robot_command_data():
    tokenizer = RobotTokenizer()
    data = []

    # More structured command templates
    command_templates = [
        # Movement commands
        {"input": "Move To Room a", "output": "Moving To Room a"},
        {"input": "Go To Room a", "output": "Moving To Room a"},
        {"input": "Move To Room b", "output": "Moving To Room b"},
        {"input": "Go To Room b", "output": "Moving To Room b"},
        {"input": "Move To Room c", "output": "Moving To Room c"},
        {"input": "Go To Room c", "output": "Moving To Room c"},
        {"input": "Move To Room d", "output": "Moving To Room d"},
        {"input": "Go To Room d", "output": "Moving To Room d"},
        
        # Cleaning commands
        {"input": "Clean Room a", "output": "Cleaning Room a"},
        {"input": "Clean Room b", "output": "Cleaning Room b"},
        {"input": "Clean Room c", "output": "Cleaning Room c"},
        {"input": "Clean Room d", "output": "Cleaning Room d"},
        
        # Home commands
        {"input": "Go Home", "output": "Moving To Home"},
        {"input": "Move Home", "output": "Moving To Home"},
        {"input": "Go To Home", "output": "Moving To Home"},
        {"input": "Move To Home", "output": "Moving To Home"},
        
        # Compound commands
        {"input": "Clean Room a And Room b", "output": "Cleaning Room a And Room b"},
        {"input": "Clean Room a And b", "output": "Cleaning Room a And Room b"},
        {"input": "Move To Room a And Clean", "output": "Moving To Room a And Cleaning"},
        {"input": "Go To Room b And Clean", "output": "Moving To Room b And Cleaning"},
        {"input": "Clean Room c And Move To Room d", "output": "Cleaning Room c And Moving To Room d"},
        {"input": "Clean Room a And b And c", "output": "Cleaning Room a And Room b And Room c"},
        {"input": "Move To Room a And b", "output": "Moving To Room a And Room b"},
        {"input": "Go To Room c And d", "output": "Moving To Room c And Room d"}
    ]

    # Generate dataset with variations
    for template in command_templates:
        # Add the original template
        data.append(template)
        
        # Variations with "The"
        input_text = template["input"]
        output_text = template["output"]
        
        # Add "The" before "Room" if not already present
        if "Room" in input_text and "The Room" not in input_text and not input_text.startswith("Go Home") and not input_text.startswith("Move Home"):
            modified_input = input_text.replace("Room", "The Room")
            if modified_input != input_text:  # Only add if actually changed
                data.append({"input": modified_input, "output": output_text})
        
        # Add more variations for better generalization
        if "And Room" in input_text:
            modified_input = input_text.replace("And Room", "And")
            data.append({"input": modified_input, "output": output_text})
        
        # Add explicit repetition of the same commands for better training
        if "Move To Room" in input_text or "Go To Room" in input_text:
            data.append({"input": input_text, "output": output_text})
            data.append({"input": input_text, "output": output_text})
        if "Clean Room" in input_text:
            data.append({"input": input_text, "output": output_text})
            data.append({"input": input_text, "output": output_text})

    # Shuffle the dataset
    random.shuffle(data)
    return data

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, decoder_output, encoder_outputs):
        batch_size = decoder_output.size(0)
        src_len = encoder_outputs.size(1)
        
        # Reshape decoder output to match encoder dimensions
        # The issue is here - we need to make sure dimensions match
        decoder_output = decoder_output.view(batch_size, 1, -1)
        decoder_output = decoder_output.expand(-1, src_len, -1)
        
        # Concatenate and compute attention scores
        attn_inputs = torch.cat((decoder_output, encoder_outputs), dim=2)
        attn_scores = self.attention(attn_inputs)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
        
        return context, attn_weights

class RobotCommandLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=3, dropout=0.3, min_length=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.min_length = min_length
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Enhanced encoder with more layers
        self.encoder = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Encoder output projection (bidirectional output to hidden_dim)
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Enhanced decoder with more layers
        self.decoder = nn.GRU(
            embedding_dim + hidden_dim,  # Concatenate embedding with context vector 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim)
        
        # Output projection layers with residual connections
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.token_attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, src, tgt=None):
        batch_size = src.size(0)
        device = src.device
        
        # Encode input sequence
        src_embedded = self.embedding(src)
        attention_weights = F.softmax(self.token_attention(src_embedded), dim=1)
        src_embedded = src_embedded * attention_weights
        encoder_outputs, encoder_hidden = self.encoder(src_embedded)
        
        # Project bidirectional encoder outputs to hidden_dim
        encoder_outputs = self.encoder_projection(encoder_outputs)
        
        # Process encoder hidden state for decoder
        # Combine bidirectional encoder hidden states
        encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        encoder_hidden = encoder_hidden.sum(dim=1)  # Sum bidirectional states
        
        if tgt is not None:
            # Training mode
            target_length = tgt.size(1)
            
            # Initialize outputs tensor
            outputs = torch.zeros(batch_size, target_length - 1, self.vocab_size).to(device)
            
            # Process all decoder inputs at once (teacher forcing)
            decoder_input = tgt[:, :-1]  # exclude last token
            decoder_embedded = self.embedding(decoder_input)
            
            # Initialize attention context
            context = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
            
            # Process each token step by step for attention
            for t in range(target_length - 1):
                # Prepare decoder input with attention context
                current_input = decoder_embedded[:, t:t+1]
                input_with_context = torch.cat([current_input, context], dim=2)
                
                # Run decoder for one step
                decoder_output, encoder_hidden = self.decoder(input_with_context, encoder_hidden)
                
                # Apply layer normalization
                decoder_output = self.layer_norm(decoder_output)
                
                # Calculate attention
                context, _ = self.attention(decoder_output, encoder_outputs)
                
                # Combine decoder output with context
                combined = torch.cat([decoder_output, context], dim=2)
                output = self.output_projection(combined)
                
                # Calculate output
                outputs[:, t] = self.fc(output).squeeze(1)
            
            return outputs
        else:
            # Inference mode
            return self.generate(src)

    def generate(self, src, max_length=30, temperature=0.5):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode input sequence
            src_embedded = self.embedding(src)
            encoder_outputs, encoder_hidden = self.encoder(src_embedded)
            
            # Project bidirectional encoder outputs to hidden_dim
            encoder_outputs = self.encoder_projection(encoder_outputs)
            
            # Process encoder hidden state for decoder
            encoder_hidden = encoder_hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
            encoder_hidden = encoder_hidden.sum(dim=1)  # Sum bidirectional states
            
            # Initialize decoder input with START token
            decoder_input = torch.full((batch_size, 1), 2, device=device)  # START token
            decoder_hidden = encoder_hidden
            
            # Initialize context vector
            context = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
            
            generated_tokens = []
            
            for i in range(max_length):
                # Get embedding for current token
                decoder_embedded = self.embedding(decoder_input)
                
                # Concatenate with context vector
                input_with_context = torch.cat([decoder_embedded, context], dim=2)
                
                # Run decoder for one step
                decoder_output, decoder_hidden = self.decoder(input_with_context, decoder_hidden)
                
                # Apply layer normalization
                decoder_output = self.layer_norm(decoder_output)
                
                # Calculate attention
                context, _ = self.attention(decoder_output, encoder_outputs)
                
                # Combine decoder output with context
                combined = torch.cat([decoder_output, context], dim=2)
                output = self.output_projection(combined)
                
                # Calculate logits
                logits = self.fc(output)
                
                # Apply temperature
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Prevent early END token
                if i < self.min_length:
                    probs[:, :, 3] = 0  # Set probability of END token to 0
                    if probs.sum() > 0:
                        probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                # Sample from probability distribution
                next_token = torch.multinomial(probs.squeeze(1), 1)
                generated_tokens.append(next_token)
                
                # Stop if END token generated after min length
                if i >= self.min_length and next_token.item() == 3:  # END token
                    break
                
                # Update decoder input for next step
                decoder_input = next_token
            
            # Concatenate all generated tokens
            return torch.cat(generated_tokens, dim=1)

class RobotCommandDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']

        input_tokens = [self.tokenizer.vocab['<START>']] + \
                       self.tokenizer.encode(input_text) + \
                       [self.tokenizer.vocab['<END>']]

        target_tokens = [self.tokenizer.vocab['<START>']] + \
                        self.tokenizer.encode(output_text) + \
                        [self.tokenizer.vocab['<END>']]

        return {
            'input': torch.LongTensor(input_tokens),
            'target': torch.LongTensor(target_tokens)
        }

def collate_fn(batch):
    max_input_len = max(item['input'].size(0) for item in batch)
    max_target_len = max(item['target'].size(0) for item in batch)

    padded_inputs = []
    padded_targets = []

    for item in batch:
        input_len = item['input'].size(0)
        target_len = item['target'].size(0)

        padded_input = torch.cat([
            item['input'],
            torch.zeros(max_input_len - input_len, dtype=torch.long)
        ])

        padded_target = torch.cat([
            item['target'],
            torch.zeros(max_target_len - target_len, dtype=torch.long)
        ])

        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)

    return {
        'input': torch.stack(padded_inputs),
        'target': torch.stack(padded_targets)
    }

def train_robot_model(model, train_loader, num_epochs=200, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    
    # Enhanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    
    # Dynamic learning rate with warm-up and decay
    initial_lr = 0.0005  # Lower initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01, betas=(0.9, 0.999))
    
    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(current_step, warmup_steps=10, total_steps=num_epochs):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 30  # Increased patience for early stopping
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input'].to(device)
            target_ids = batch['target'].to(device)
            
            # Clear previous gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, target_ids)
            
            # Reshape outputs and targets for loss calculation
            targets = target_ids[:, 1:]  # Exclude START token
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            
            # Apply label smoothing
            smoothing = 0.1
            n_classes = outputs_flat.size(-1)
            
            # Calculate loss with label smoothing
            loss = criterion(outputs_flat, targets_flat)
            
            # Apply masking for padded tokens
            mask = (targets_flat != 0).float()
            loss = (loss * mask).sum() / mask.sum()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(train_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            print(f"New best model saved! Loss: {best_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_loss

import math  # Add import for math functions

def save_best_model(model, tokenizer, save_path='robot_llm_model.pth'):
    # Save the model state dict and tokenizer vocab
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'id2word': tokenizer.id2word
    }, save_path)
    
    print(f"Best model saved to {save_path}")

def load_robot_model(model_path='robot_llm_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize tokenizer
    tokenizer = RobotTokenizer()
    tokenizer.vocab = checkpoint['vocab']
    tokenizer.id2word = checkpoint['id2word']
    
    # Initialize model
    model = RobotCommandLLM(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=128,
        hidden_dim=512,
        num_layers=3,
        dropout=0.3,
        min_length=3
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def generate_response(model, tokenizer, input_text, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    with torch.no_grad():
        # Normalize input text
        input_text = ' '.join(input_text.split())  # Remove extra spaces
        
        # Tokenize input
        tokens = tokenizer.encode(input_text)
        tokens = [tokenizer.vocab['<START>']] + tokens
        input_tensor = torch.LongTensor([tokens]).to(device)
        
        # Generate response
        output_tokens = model.generate(input_tensor, temperature=0.6)
        response = tokenizer.decode(output_tokens[0].tolist())
        
        # Print the execution (as specified in requirements)
        print(f"Executing: {response}")
        
        return response