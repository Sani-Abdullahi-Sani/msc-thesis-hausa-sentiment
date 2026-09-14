"""
Transformer model architecture module.
Provides the transformer model implementation for text classification.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to the input embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    """Transformer model for text classification"""
    def __init__(self, vocab_size, num_labels, embed_dim=256, num_heads=16,
                 num_layers=4, ff_dim=512, dropout=0.1,
                 init_method='xavier', beta=2.0, multilabel=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.init_method = init_method
        self.beta = beta
        self.multilabel = multilabel

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_labels)

        # Initialize weights using the specified method
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize model weights based on specified method.
        
        Args:
            module: Module to initialize
        """
        from src.models.init_methods import xavier_init, he_init, generalized_init
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if self.init_method == 'xavier':
                xavier_init(module.weight, gain=1.0)
            elif self.init_method == 'he':
                he_init(module.weight, gain=1.0)
            elif self.init_method == 'generalized':
                generalized_init(module.weight, beta=self.beta, gain=1.0)
            else:
                raise ValueError(f"Unknown initialization method: {self.init_method}")

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the transformer classifier.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            attention_mask: Attention mask of shape [batch_size, seq_len]
            
        Returns:
            Logits of shape [batch_size, num_labels]
        """
        # Create padding mask for transformer
        padding_mask = (attention_mask == 0)

        # Embedding and positional encoding
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoder(embedded)

        # Apply transformer encoder
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)

        # Global pooling (mean pooling over non-padding tokens)
        mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size())
        sum_embeddings = torch.sum(encoded * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        pooled_output = sum_embeddings / sum_mask

        # Classification
        logits = self.classifier(pooled_output)

        return logits

def collate_fn(batch):
    """
    Collate function for DataLoader that handles variable length sequences.
    Pads sequences and converts labels to tensors.
    
    Args:
        batch: Batch of samples
        
    Returns:
        Dictionary with padded inputs and labels
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_ids and attention_masks
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Handle labels
    if isinstance(labels[0], torch.Tensor) and labels[0].dtype == torch.float:  # Multilabel
        # Pad labels with -1 to the maximum sequence length
        max_len = max(label.shape[0] for label in labels)
        padded_labels = [
            torch.cat([label, torch.full((max_len - label.shape[0],), -1, dtype=label.dtype)])
            for label in labels
        ]
        labels = torch.stack(padded_labels)
    else:  # Single-label
        labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'labels': labels
    }