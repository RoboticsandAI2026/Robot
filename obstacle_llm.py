import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import gc


class SimpleTokenizer:
    def __init__(self):
        # Vocabulary focusing on core concepts and natural language
        self.vocab = {
            # Special tokens
            '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3,
            # Template tokens
            '<OBJ>': 4, '<ACTION>': 5, '<MATERIAL>': 6, '<PURPOSE>': 7,
            # Objects
            'water': 8, 'bottle': 9, 'shoe': 10, 'cap': 11,
            'dust': 12, 'pan': 13, 'football': 14,
            # Actions and verbs
            'carry': 15, 'store': 16, 'protect': 17, 'cover': 18,
            'hold': 19, 'drink': 20, 'shield': 21, 'play': 22,
            'designed': 23, 'used': 24, 'made': 25, 'helps': 26,
            'provides': 27, 'allows': 28,
            # Materials
            'plastic': 29, 'metal': 30, 'leather': 31, 'fabric': 32,
            'rubber': 33,
            # Purpose/descriptors
            'liquid': 34, 'water': 35, 'feet': 36, 'rain': 37,
            'sports': 38, 'game': 39, 'protective': 40, 'portable': 41,
            'container': 42, 'equipment': 43, 'footwear': 44, 'head': 45, 
            'sun': 46, 'household': 47, 'collect': 48,
            # Structure words
            'a': 49, 'an': 50, 'is': 51, 'to': 52, 'for': 53,
            'and': 54, 'or': 55, 'from': 56, 'with': 57, 'in': 58,
            'that': 59, 'which': 60, 'while': 61, 'during': 62,
            'against': 63, 'typically': 64, 'commonly': 65, 'specifically': 66,
            'the': 67, 'this': 68, 'these': 69, 'those': 70,
            # Additional descriptive words
            'comfortable': 71, 'durable': 72, 'useful': 73, 'convenient': 74,
            'essential': 75, 'practical': 76, 'popular': 77, 'object': 78, 'under': 79,
            # Add extra token to ensure vocab_size > max_token_id
            '<EXTRA>': 80
        }

        self.id2word = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in text.lower().split()]

    def decode(self, tokens):
        words = []
        for token in tokens:
            if token in [self.vocab['<PAD>'], self.vocab['<START>'], self.vocab['<END>'], self.vocab['<EXTRA>']]:
                continue
            word = self.id2word.get(token, '<UNK>')
            if word.startswith('<') and word.endswith('>'):
                continue
            words.append(word)

        if not words:
            return "No response generated."

        # Basic sentence cleanup
        words[0] = words[0].capitalize()
        return ' '.join(words) + '.'


def create_obstacle_data():
    tokenizer = SimpleTokenizer()
    data = []

    # Only using the object name as the key identifier, treating other question variations similarly
    base_questions = [
        "what is {}",
        "explain {}",
        "describe {}",
        "tell me about {}",
        "how would you describe {}",
        "can you tell me about {}"
    ]

    # Object-specific response patterns
    templates = {
        'water bottle': [
            "a water bottle is a portable container that helps carry and store water it is typically made of plastic or metal",
            "a water bottle is a container designed for carrying liquids made from plastic or metal materials",
            "a water bottle is used to store and transport water it is made of plastic or metal"
        ],
        'shoe': [
            "a shoe is protective footwear that covers and protects the feet it is made of leather or fabric",
            "a shoe is footwear designed to protect feet while walking made from leather or fabric",
            "a shoe is used to protect and cover feet it is made of leather or fabric"
        ],
        'cap': [
            "a cap is protective wear that covers and protects the head it is made of leather or fabric",
            "a cap is object designed to protect head while walking under sun made from leather or fabric",
            "a cap is used to protect and cover head it is made of leather or fabric"
        ],
        'dust pan': [
            "a dust pan is a household object that helps carry the dust it is typically made of plastic or metal",
            "a dust pan is designed to collect and carry dust it is made with fabric and metal",
            "a dust pan is used to carry dust it is made of fabric and metal"
        ],
        'football': [
            "a football is sports equipment made of leather or rubber used for playing games",
            "a football is designed for playing sports it is made of leather or rubber",
            "a football is used to play sports and games it is made of leather or rubber"
        ]
    }

    # Process each object
    for obj, responses in templates.items():
        # Create variations of the object name for questions
        obj_variations = [
            obj,
            f"the {obj}",
            f"a {obj}",
            f"purpose of {obj}",
            f"purpose of the {obj}"
        ]

        # Generate question-response pairs
        for obj_var in obj_variations:
            for question in base_questions:
                full_question = question.format(obj_var)
                for response in responses:
                    data.append({
                        "input": full_question,
                        "output": response
                    })

    random.shuffle(data)
    return data


class SimpleObstacleLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, min_length=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        
        # Add min_length parameter
        self.min_length = min_length
        
        # Ensure consistent dimensions by storing the hidden_dim
        self.hidden_dim = hidden_dim

        # Attention components with explicit dimension specification
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, vocab_size)
        )

    # Replace your current attention_mechanism method with this improved version
    def attention_mechanism(self, decoder_output, encoder_outputs):
        """
        Improved attention mechanism with better dimension handling and error checking
        """
        # Ensure decoder output is properly shaped [batch_size, 1, hidden_dim]
        if decoder_output.dim() == 2:
            decoder_output = decoder_output.unsqueeze(1)
        
        # Get dimensions
        batch_size = decoder_output.size(0)
        src_len = encoder_outputs.size(1)
        decoder_hidden_size = decoder_output.size(2)
        encoder_hidden_size = encoder_outputs.size(2)
        
        # Check for NaN/Inf in input tensors
        if torch.isnan(decoder_output).any() or torch.isinf(decoder_output).any():
            print("Warning: NaN/Inf detected in decoder_output")
            # Replace with small values
            decoder_output = torch.where(torch.isnan(decoder_output) | torch.isinf(decoder_output), 
                                        torch.tensor(1e-5, device=decoder_output.device), 
                                        decoder_output)
        
        if torch.isnan(encoder_outputs).any() or torch.isinf(encoder_outputs).any():
            print("Warning: NaN/Inf detected in encoder_outputs")
            encoder_outputs = torch.where(torch.isnan(encoder_outputs) | torch.isinf(encoder_outputs), 
                                        torch.tensor(1e-5, device=encoder_outputs.device), 
                                        encoder_outputs)
        
        # Instead of creating projections on the fly, use a consistent approach
        if not hasattr(self, 'dim_projection') and decoder_hidden_size != encoder_hidden_size:
            # Create the projection only once and add it as a module
            self.dim_projection = nn.Linear(decoder_hidden_size, encoder_hidden_size).to(decoder_output.device)
            # Make sure it's registered as a module
            setattr(self, f'dim_projection_{decoder_hidden_size}_{encoder_hidden_size}', self.dim_projection)
        
        # Repeat decoder output for each encoder position
        decoder_output = decoder_output.repeat(1, src_len, 1)
        
        # Apply projection if necessary
        if decoder_hidden_size != encoder_hidden_size:
            if hasattr(self, 'dim_projection'):
                decoder_output = self.dim_projection(decoder_output)
            else:
                # Handle case where dimensions don't match and we don't have a projection yet
                print(f"Warning: Dimension mismatch - decoder: {decoder_hidden_size}, encoder: {encoder_hidden_size}")
                # Use a simple approach - just take or add dimensions as needed
                if decoder_hidden_size < encoder_hidden_size:
                    padding = torch.zeros(batch_size, src_len, encoder_hidden_size - decoder_hidden_size, 
                                        device=decoder_output.device)
                    decoder_output = torch.cat([decoder_output, padding], dim=2)
                else:
                    decoder_output = decoder_output[:, :, :encoder_hidden_size]
        
        # Now concatenate along the feature dimension
        attn_inputs = torch.cat((decoder_output, encoder_outputs), dim=2)
        
        # Apply attention scores with careful error checking
        try:
            attn_scores = self.attention(attn_inputs)
            
            # Check for NaN/Inf
            if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
                print("Warning: NaN/Inf in attention scores, replacing with zeros")
                attn_scores = torch.zeros_like(attn_scores)
                
            # Apply softmax with careful handling
            attn_weights = F.softmax(attn_scores, dim=1)
            
            # Check weights again
            if torch.isnan(attn_weights).any():
                print("Warning: NaN in attention weights after softmax")
                attn_weights = torch.ones_like(attn_weights) / attn_weights.size(1)
                
            # Ensure weights sum to 1
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Apply attention
            context = torch.bmm(attn_weights.transpose(1, 2), encoder_outputs)
            
        except Exception as e:
            print(f"Error in attention calculation: {e}")
            print(f"attn_inputs shape: {attn_inputs.shape}")
            # Fallback: use average of encoder outputs as context
            context = encoder_outputs.mean(dim=1, keepdim=True)
        
        return context

    def forward(self, src, tgt=None):
        src_embedded = self.embedding(src)
        encoder_outputs, encoder_hidden = self.encoder(src_embedded)

        if tgt is not None:
            # Training mode
            batch_size = tgt.size(0)
            target_length = tgt.size(1)
            vocab_size = self.fc[-1].out_features

            # Initialize outputs tensor
            outputs = torch.zeros(batch_size, target_length - 1, vocab_size).to(src.device)

            # Teacher forcing - use target tokens as input
            decoder_input = tgt[:, :-1]  # exclude last token
            decoder_embedded = self.embedding(decoder_input)

            # Initial decoder hidden state
            decoder_hidden = encoder_hidden

            # Process whole sequence at once
            decoder_outputs, _ = self.decoder(decoder_embedded, decoder_hidden)

            # Apply attention for each time step
            for t in range(target_length - 1):
                decoder_output = decoder_outputs[:, t:t+1]
                try:
                    context = self.attention_mechanism(decoder_output, encoder_outputs)
                    combined = torch.cat((decoder_output, context), dim=2)
                    output = self.output_projection(combined)
                    outputs[:, t] = self.fc(output).squeeze(1)
                except Exception as e:
                    print(f"Error at time step {t}:")
                    print(f"decoder_output shape: {decoder_output.shape}")
                    print(f"encoder_outputs shape: {encoder_outputs.shape}")
                    raise e

            return outputs
        else:
            return self.generate(src)

    def generate(self, src, max_length=50):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device

            # Encode input sequence
            src_embedded = self.embedding(src)
            encoder_outputs, encoder_hidden = self.encoder(src_embedded)

            # Initialize decoder input with START token
            decoder_input = torch.full((batch_size, 1), 2, device=device)  # START token
            decoder_hidden = encoder_hidden

            generated_tokens = []
            for i in range(max_length):
                decoder_embedded = self.embedding(decoder_input)
                decoder_output, decoder_hidden = self.decoder(decoder_embedded, decoder_hidden)
                
                # Apply layer normalization
                decoder_output = self.layer_norm(decoder_output)
                
                context = self.attention_mechanism(decoder_output, encoder_outputs)
                combined = torch.cat((decoder_output, context), dim=2)
                output = self.output_projection(combined)
                output = self.fc(output)
                
                # Temperature sampling
                temperature = 0.5
                output = output / temperature
                probs = F.softmax(output, dim=-1)
                
                # Prevent early END token
                if i < self.min_length:
                    probs[:, :, 3] = 0  # Set probability of END token to 0
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

                # Sample from probability distribution
                next_token = torch.multinomial(probs.squeeze(1), 1)
                generated_tokens.append(next_token)

                # Only end generation if we've exceeded minimum length
                if i >= self.min_length and next_token.item() == 3:  # END token
                    break

                decoder_input = next_token

            return torch.cat(generated_tokens, dim=1)


class ObstacleDataset(Dataset):
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


# Add these checks to your train_model function before the model forward pass
# Place this right after the line: input_ids = batch['input'].to(device)

# Replace your current train_model function with this safer implementation

def train_model(model, train_loader, num_epochs=150, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Get vocab size directly from model's embedding layer
    vocab_size = model.embedding.num_embeddings
    print(f"Model vocabulary size: {vocab_size}")

    best_loss = float('inf')
    patience = 0
    max_patience = 15

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input'].to(device)
            target_ids = batch['target'].to(device)
            
            # First, clamp all indices to be within vocabulary range
            # This prevents out-of-bounds errors
            input_ids = torch.clamp(input_ids, 0, vocab_size-1)
            target_ids = torch.clamp(target_ids, 0, vocab_size-1)
            
            optimizer.zero_grad()
            
            try:
                outputs = model(input_ids, target_ids)
                
                # Get the prediction targets (excluding the first token)
                targets = target_ids[:, 1:]
                
                # Reshape outputs and targets for loss calculation
                batch_size = outputs.size(0)
                seq_length = outputs.size(1)
                outputs_flat = outputs.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                
                # Apply a safer label smoothing approach
                smoothing = 0.1
                
                # Create mask for non-padding tokens
                non_pad_mask = (targets_flat != 0).float()
                
                # Calculate standard cross-entropy loss
                loss = F.cross_entropy(
                    outputs_flat, 
                    targets_flat, 
                    ignore_index=0,  # Ignore padding
                    reduction='none'
                )
                
                # Apply the loss only to non-padding tokens
                loss = (loss * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1e-5)
                
                # Apply label smoothing manually in a safer way
                if smoothing > 0:
                    # Get one-hot targets
                    one_hot = torch.zeros_like(outputs_flat).to(device)
                    
                    # Only scatter for valid indices
                    valid_indices = (targets_flat < vocab_size) & (targets_flat >= 0) & (targets_flat != 0)
                    
                    # Scatter operation only for valid indices
                    if valid_indices.any():
                        # Get the valid targets
                        valid_targets = targets_flat[valid_indices]
                        
                        # Create index tensor for scatter
                        index_tensor = torch.zeros(valid_indices.sum(), 1, dtype=torch.long, device=device)
                        index_tensor[:, 0] = valid_targets
                        
                        # Select only valid outputs
                        valid_outputs = outputs_flat[valid_indices]
                        
                        # Create one-hot for valid outputs
                        valid_one_hot = torch.zeros_like(valid_outputs).to(device)
                        valid_one_hot.scatter_(1, index_tensor, 1)
                        
                        # Apply smoothing
                        valid_one_hot = valid_one_hot * (1 - smoothing) + smoothing / vocab_size
                        
                        # Calculate KL divergence loss
                        log_probs = F.log_softmax(valid_outputs, dim=-1)
                        smoothed_loss = -(valid_one_hot * log_probs).sum(dim=-1).mean()
                        
                        # Combine with regular loss
                        loss = (1 - smoothing) * loss + smoothing * smoothed_loss
                
                loss.backward()
                
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
            except Exception as e:
                print(f"Error during training: {e}")
                print(f"Input shape: {input_ids.shape}, max value: {input_ids.max().item()}")
                print(f"Target shape: {target_ids.shape}, max value: {target_ids.max().item()}")
                print(f"Vocab size: {vocab_size}")
                raise e

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1

        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break