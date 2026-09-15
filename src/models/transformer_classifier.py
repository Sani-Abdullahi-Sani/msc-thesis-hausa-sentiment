import math
import torch
import torch.nn as nn
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
    """
    Transformer model for text classification with support for multiple initialization methods.
    
    Supports Xavier, He, Generalized, and Imbalanced initialization methods.
    """
    
    def __init__(self, vocab_size, num_labels, embed_dim=256, num_heads=16,
                 num_layers=4, ff_dim=512, dropout=0.1,
                 init_method='xavier', beta=2.0, beta_input=1.0, beta_output=3.0,
                 multilabel=False):
        super().__init__()
        
        # Store model configuration
        self.vocab_size = vocab_size
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.multilabel = multilabel
        
        # Store initialization parameters
        self.init_method = init_method
        self.beta = beta  # For backward compatibility with uniform generalized
        self.beta_input = beta_input  # For imbalanced initialization
        self.beta_output = beta_output  # For imbalanced initialization
        
        print(f"🏗️  Creating TransformerClassifier:")
        print(f"   📊 Vocab size: {vocab_size}")
        print(f"   🎯 Labels: {num_labels} ({'multilabel' if multilabel else 'single-label'})")
        print(f"   🧠 Architecture: embed_dim={embed_dim}, heads={num_heads}, layers={num_layers}")
        print(f"   🎛️  Dropout: {dropout}")
        print(f"   ⚡ Initialization: {init_method}")
        
        if init_method == 'generalized':
            print(f"   📐 Beta: {beta}")
        elif init_method == 'generalized_imbalanced':
            print(f"   📐 Beta input→output: {beta_input}→{beta_output} (ratio: {beta_output/beta_input:.1f}x)")

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

        # Apply initialization
        self._apply_initialization()

    def _apply_initialization(self):
        """Apply the specified initialization method to the model."""
        print(f"\n🔧 Applying {self.init_method} initialization...")
        
        from .initialization import apply_initialization
        
        # Apply initialization and get summary
        init_summary = apply_initialization(
            model=self,
            init_method=self.init_method,
            beta=self.beta,
            beta_input=self.beta_input,
            beta_output=self.beta_output,
            gain=1.0
        )
        
        # Print initialization summary
        print(f"✅ Initialization complete!")
        if self.init_method == 'generalized_imbalanced':
            print(f"   📊 Initialized {init_summary['total_layers']} layers:")
            print(f"   📥 Input layers: {init_summary['input_layers']} (β={init_summary['beta_input']})")
            print(f"   🔄 Middle layers: {init_summary['middle_layers']} (β={(init_summary['beta_input'] + init_summary['beta_output'])/2:.1f})")
            print(f"   📤 Output layers: {init_summary['output_layers']} (β={init_summary['beta_output']})")
            print(f"   ⚖️  Imbalance ratio: {init_summary['imbalance_ratio']:.1f}x")
        else:
            print(f"   📊 Initialized {init_summary['total_layers']} layers with {init_summary['method']}")
            if 'beta' in init_summary and init_summary['beta'] is not None:
                print(f"   📐 Beta: {init_summary['beta']}")

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
