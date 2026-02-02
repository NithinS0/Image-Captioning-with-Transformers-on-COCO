"""
train.py - Training Loop

Train image captioning model with teacher forcing.
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

import config
from vocab import build_vocab, save_vocab, load_vocab
from dataset import get_dataloader
from encoder import Encoder
from decoder import Decoder


class ImageCaptioningModel(nn.Module):
    """Complete image captioning model."""
    
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        self.encoder = Encoder(embed_dim=embed_dim, pretrained=True, fine_tune=False)
        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, images, captions):
        """
        Forward pass.
        
        Args:
            images: (batch, 3, 224, 224)
            captions: (batch, seq_len)
        
        Returns:
            predictions: (batch, seq_len, vocab_size)
        """
        encoder_out = self.encoder(images)
        predictions = self.decoder(captions, encoder_out)
        return predictions
    
    def generate_caption(self, image, vocab, max_length=50, device='cuda'):
        """Generate caption for a single image."""
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(image)
            caption_words = self.decoder.generate_caption(encoder_out, vocab, max_length, device)
        return caption_words


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, num_epochs):
    """
    Train for one epoch.
    
    Args:
        model: ImageCaptioningModel
        dataloader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: Device
        epoch: Current epoch
        num_epochs: Total epochs
    
    Returns:
        avg_loss: Average loss for epoch
    """
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (images, captions, lengths) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        captions = captions.to(device)
        
        # Teacher forcing:
        # Input: all tokens except last  → <START> a cat sitting
        # Target: all tokens except first → a cat sitting <END>
        decoder_input = captions[:, :-1]
        targets = captions[:, 1:]
        
        # Mixed precision forward pass
        if config.USE_MIXED_PRECISION:
            with autocast():
                predictions = model(images, decoder_input)
                
                # Reshape for CrossEntropyLoss
                batch_size, seq_len, vocab_size = predictions.shape
                predictions = predictions.reshape(-1, vocab_size)
                targets = targets.reshape(-1)
                
                loss = criterion(predictions, targets)
        else:
            predictions = model(images, decoder_input)
            batch_size, seq_len, vocab_size = predictions.shape
            predictions = predictions.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        
        if config.USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
            optimizer.step()
        
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f'  Epoch [{epoch}/{num_epochs}] '
                  f'Batch [{batch_idx + 1}/{len(dataloader)}] '
                  f'Loss: {avg_loss:.4f} '
                  f'Time: {elapsed:.1f}s')
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f'  ✅ Checkpoint saved: {filepath}')


def main():
    """Main training function."""
    
    print("="*70)
    print("Image Captioning Training")
    print("="*70)
    
    # Display configuration
    config.display_config()
    
    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Device
    device = config.DEVICE
    print(f"\n✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== Build/Load Vocabulary ==========
    print("\n" + "="*70)
    print("Vocabulary")
    print("="*70)
    
    if os.path.exists(config.VOCAB_PATH):
        vocab = load_vocab(config.VOCAB_PATH)
    else:
        vocab = build_vocab(
            config.TRAIN_ANNOTATIONS,
            threshold=config.VOCAB_THRESHOLD,
            max_samples=config.MAX_VOCAB_SAMPLES
        )
        save_vocab(vocab, config.VOCAB_PATH)
    
    print(f"✅ Vocabulary size: {len(vocab)}")
    
    # ========== Create DataLoader ==========
    print("\n" + "="*70)
    print("DataLoader")
    print("="*70)
    
    train_loader = get_dataloader(
        root=config.TRAIN_IMAGES,
        annotation_file=config.TRAIN_ANNOTATIONS,
        vocab=vocab,
        batch_size=config.BATCH_SIZE,
        is_train=True,
        max_samples=config.MAX_TRAIN_SAMPLES
    )
    
    print(f"✅ Training batches: {len(train_loader)}")
    
    # ========== Create Model ==========
    print("\n" + "="*70)
    print("Model")
    print("="*70)
    
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # ========== Loss and Optimizer ==========
    pad_idx = vocab.word2idx[vocab.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    print(f"\n✅ Loss: CrossEntropyLoss (ignore_index={pad_idx})")
    print(f"✅ Optimizer: Adam (lr={config.LEARNING_RATE})")
    print(f"✅ Mixed precision: {config.USE_MIXED_PRECISION}")
    
    # ========== Training Loop ==========
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    
    best_loss = float('inf')
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 70)
        
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, config.NUM_EPOCHS
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Loss: {train_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s ({epoch_time/60:.1f}m)")
        
        # Save checkpoint
        is_best = train_loss < best_loss
        
        if is_best:
            best_loss = train_loss
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            )
            print(f"   🏆 New best model!")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, train_loss,
                os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # ========== Training Complete ==========
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"✅ Best loss: {best_loss:.4f}")
    print(f"✅ Checkpoints saved in: {config.CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()
