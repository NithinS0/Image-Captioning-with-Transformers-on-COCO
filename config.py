"""
config.py - Hyperparameters and Configuration

All configuration settings in one place.
"""

import os

# ============================================================================
# PATHS
# ============================================================================

# Data paths
DATA_ROOT = './data/coco'
TRAIN_IMAGES = os.path.join(DATA_ROOT, 'images/val2017')
TRAIN_ANNOTATIONS = os.path.join(DATA_ROOT, 'annotations/captions_val2017.json')
VAL_IMAGES = os.path.join(DATA_ROOT, 'images/val2017')
VAL_ANNOTATIONS = os.path.join(DATA_ROOT, 'annotations/captions_val2017.json')

# Output paths
CHECKPOINT_DIR = './checkpoints'
VOCAB_PATH = os.path.join(CHECKPOINT_DIR, 'vocab.pkl')

# ============================================================================
# DATASET
# ============================================================================

# Dataset limits (for memory optimization)
MAX_TRAIN_SAMPLES = 500      # Limit training samples (None = all)
MAX_VAL_SAMPLES = 100        # Limit validation samples
MAX_VOCAB_SAMPLES = 5000     # Limit samples for vocab building

# Vocabulary
VOCAB_THRESHOLD = 3          # Minimum word frequency
MAX_CAPTION_LENGTH = 50      # Maximum caption length

# Image preprocessing
IMAGE_SIZE = 224             # Image size (224x224)
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
IMAGE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# ============================================================================
# MODEL
# ============================================================================

# Encoder (ResNet50)
ENCODER_PRETRAINED = True    # Use pretrained ResNet50
ENCODER_FINE_TUNE = False    # Freeze encoder initially

# Decoder (Transformer)
EMBED_DIM = 256              # Embedding dimension (256 for limited GPU)
NUM_HEADS = 8                # Number of attention heads
NUM_LAYERS = 3               # Number of transformer layers (3 for limited GPU)
DIM_FEEDFORWARD = 1024       # Feedforward dimension
DROPOUT = 0.1                # Dropout rate

# ============================================================================
# TRAINING
# ============================================================================

# Training parameters
BATCH_SIZE = 8               # Batch size (small for limited GPU)
NUM_EPOCHS = 5               # Number of epochs
LEARNING_RATE = 1e-4         # Learning rate

# Optimization
GRAD_CLIP = 5.0              # Gradient clipping max norm
USE_MIXED_PRECISION = True   # Use FP16 mixed precision

# Logging
LOG_INTERVAL = 50            # Log every N batches

# ============================================================================
# EVALUATION
# ============================================================================

# Evaluation parameters
EVAL_BATCH_SIZE = 1          # Batch size for evaluation
MAX_DECODE_LENGTH = 50       # Maximum caption length during generation
NUM_EVAL_SAMPLES = 10        # Number of samples to evaluate

# ============================================================================
# DEVICE
# ============================================================================

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DISPLAY CONFIG
# ============================================================================

def display_config():
    """Display configuration."""
    print("="*70)
    print("Configuration")
    print("="*70)
    print(f"\n📁 Paths:")
    print(f"  Train images: {TRAIN_IMAGES}")
    print(f"  Train annotations: {TRAIN_ANNOTATIONS}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}")
    
    print(f"\n📊 Dataset:")
    print(f"  Max train samples: {MAX_TRAIN_SAMPLES}")
    print(f"  Max val samples: {MAX_VAL_SAMPLES}")
    print(f"  Vocab threshold: {VOCAB_THRESHOLD}")
    print(f"  Max caption length: {MAX_CAPTION_LENGTH}")
    
    print(f"\n🤖 Model:")
    print(f"  Embed dim: {EMBED_DIM}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Num layers: {NUM_LAYERS}")
    print(f"  Dim feedforward: {DIM_FEEDFORWARD}")
    
    print(f"\n🎯 Training:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Mixed precision: {USE_MIXED_PRECISION}")
    
    print(f"\n💻 Device: {DEVICE}")
    print("="*70)


if __name__ == '__main__':
    display_config()
