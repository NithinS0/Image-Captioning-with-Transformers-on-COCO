"""
decoder.py - Transformer Decoder

Transformer decoder for caption generation using PyTorch's built-in TransformerDecoder.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix: (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding.
        
        Args:
            x: (batch, seq_len, embed_dim)
        
        Returns:
            x with positional encoding: (batch, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Decoder(nn.Module):
    """
    Transformer Decoder using PyTorch's built-in nn.TransformerDecoder.
    
    Input:  captions (batch, seq_len)
            encoder_out (batch, 49, embed_dim)
    Output: predictions (batch, seq_len, vocab_size)
    """
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # IMPORTANT: batch_first=True
        )
        
        # Stack decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, captions, encoder_out):
        """
        Forward pass.
        
        Args:
            captions: (batch, seq_len) - token indices
            encoder_out: (batch, 49, embed_dim) - image features
        
        Returns:
            predictions: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = captions.shape
        
        # Embed tokens: (batch, seq_len) → (batch, seq_len, embed_dim)
        embeddings = self.embedding(captions)
        
        # Add positional encoding: (batch, seq_len, embed_dim)
        embeddings = self.pos_encoder(embeddings)
        
        # Create causal mask: (seq_len, seq_len)
        # Prevents attending to future tokens
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(captions.device)
        
        # Transformer decoder
        # tgt: (batch, seq_len, embed_dim) - target sequence (captions)
        # memory: (batch, 49, embed_dim) - encoder output (image features)
        decoder_out = self.transformer_decoder(
            tgt=embeddings,
            memory=encoder_out,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary: (batch, seq_len, embed_dim) → (batch, seq_len, vocab_size)
        predictions = self.fc_out(decoder_out)
        
        return predictions
    
    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask.
        
        Args:
            sz: Sequence length
        
        Returns:
            mask: (sz, sz) with -inf in upper triangle
        
        Example for sz=4:
        [[0,    -inf, -inf, -inf],
         [0,    0,    -inf, -inf],
         [0,    0,    0,    -inf],
         [0,    0,    0,    0   ]]
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def generate_caption(self, encoder_out, vocab, max_length=50, device='cuda'):
        """
        Generate caption using greedy decoding.
        
        Args:
            encoder_out: (1, 49, embed_dim) - single image features
            vocab: Vocabulary object
            max_length: Maximum caption length
            device: Device to run on
        
        Returns:
            caption_words: List of words
        """
        self.eval()
        
        with torch.no_grad():
            # Start with <START> token
            caption = [vocab.word2idx[vocab.start_token]]
            
            for _ in range(max_length):
                # Convert to tensor: (1, current_length)
                caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
                
                # Get predictions: (1, current_length, vocab_size)
                predictions = self.forward(caption_tensor, encoder_out)
                
                # Get last token prediction: (vocab_size,)
                last_pred = predictions[0, -1, :]
                
                # Greedy: take argmax
                predicted_idx = last_pred.argmax().item()
                
                # Add to caption
                caption.append(predicted_idx)
                
                # Stop if <END> token
                if predicted_idx == vocab.word2idx[vocab.end_token]:
                    break
            
            # Convert indices to words (exclude <START> and <END>)
            caption_words = []
            for idx in caption[1:-1]:  # Skip <START> and <END>
                word = vocab.idx2word.get(idx, vocab.unk_token)
                if word != vocab.pad_token:
                    caption_words.append(word)
        
        return caption_words


if __name__ == '__main__':
    # Test decoder
    vocab_size = 10000
    decoder = Decoder(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=3
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    
    captions = torch.randint(0, vocab_size, (batch_size, seq_len))
    encoder_out = torch.randn(batch_size, 49, 256)
    
    predictions = decoder(captions, encoder_out)
    
    print(f"Input captions: {captions.shape}")
    print(f"Input encoder_out: {encoder_out.shape}")
    print(f"Output predictions: {predictions.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
