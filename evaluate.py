"""
evaluate.py - Caption Generation & BLEU Score

Evaluate trained model with BLEU scores and visualization.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math

import config
from vocab import load_vocab
from dataset import get_dataloader, get_transform
from encoder import Encoder
from decoder import Decoder
from train import ImageCaptioningModel


def calculate_bleu(reference, candidate, n=4):
    """
    Calculate BLEU score.
    
    Args:
        reference: List of reference words
        candidate: List of candidate words
        n: Maximum n-gram (4 for BLEU-4)
    
    Returns:
        BLEU score (0 to 1)
    """
    # Calculate n-gram precisions
    precisions = []
    
    for i in range(1, n + 1):
        # Get n-grams
        ref_ngrams = Counter([tuple(reference[j:j+i]) 
                              for j in range(len(reference) - i + 1)])
        cand_ngrams = Counter([tuple(candidate[j:j+i]) 
                               for j in range(len(candidate) - i + 1)])
        
        if len(cand_ngrams) == 0:
            precisions.append(0)
            continue
        
        # Clipped counts
        clipped = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total = sum(cand_ngrams.values())
        
        precision = clipped / total if total > 0 else 0
        precisions.append(precision)
    
    # Brevity penalty
    ref_len = len(reference)
    cand_len = len(candidate)
    
    if cand_len > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / cand_len) if cand_len > 0 else 0
    
    # BLEU score
    if min(precisions) > 0:
        log_precisions = sum(math.log(p) for p in precisions) / n
        bleu = bp * math.exp(log_precisions)
    else:
        bleu = 0
    
    return bleu


def denormalize_image(image_tensor):
    """
    Denormalize image for display.
    
    Args:
        image_tensor: (3, H, W) normalized tensor
    
    Returns:
        image: (H, W, 3) numpy array [0, 1]
    """
    mean = torch.tensor(config.IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGE_STD).view(3, 1, 1)
    
    img = image_tensor * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    
    return img


def display_sample(image, generated, ground_truth, bleu, sample_num):
    """
    Display image with captions.
    
    Args:
        image: Image tensor (3, 224, 224)
        generated: List of generated words
        ground_truth: List of ground truth words
        bleu: BLEU score
        sample_num: Sample number
    """
    img = denormalize_image(image)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    
    title = f"Sample {sample_num}\n\n"
    title += f"Generated: {' '.join(generated)}\n\n"
    title += f"Ground Truth: {' '.join(ground_truth)}\n\n"
    title += f"BLEU-4: {bleu:.4f}"
    
    plt.title(title, fontsize=12, wrap=True, pad=20)
    plt.tight_layout()
    plt.show()


def evaluate(model, dataset, vocab, device, num_samples=10):
    """
    Evaluate model on samples.
    
    Args:
        model: Trained model
        dataset: Dataset
        vocab: Vocabulary
        device: Device
        num_samples: Number of samples to evaluate
    
    Returns:
        avg_bleu: Average BLEU score
    """
    model.eval()
    
    bleu_scores = []
    
    print("="*70)
    print(f"Evaluating {num_samples} samples with Greedy Decoding")
    print("="*70)
    
    for i in range(min(num_samples, len(dataset))):
        # Get sample
        image, caption, length = dataset[i]
        
        # Generate caption
        image_batch = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_words = model.generate_caption(
                image_batch, vocab, max_length=config.MAX_DECODE_LENGTH, device=device
            )
        
        # Get ground truth
        gt_words = []
        for idx in caption.tolist():
            word = vocab.idx2word[idx]
            if word == vocab.end_token:
                break
            if word not in [vocab.start_token, vocab.pad_token]:
                gt_words.append(word)
        
        # Calculate BLEU
        bleu = calculate_bleu(gt_words, generated_words, n=4)
        bleu_scores.append(bleu)
        
        # Print
        print(f"\nSample {i+1}:")
        print(f"  Generated:    {' '.join(generated_words)}")
        print(f"  Ground Truth: {' '.join(gt_words)}")
        print(f"  BLEU-4:       {bleu:.4f}")
        
        # Display
        display_sample(image, generated_words, gt_words, bleu, i+1)
    
    # Calculate average
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)
    print(f"Samples evaluated: {len(bleu_scores)}")
    print(f"Average BLEU-4: {avg_bleu:.4f}")
    print(f"Min BLEU-4: {min(bleu_scores):.4f}")
    print(f"Max BLEU-4: {max(bleu_scores):.4f}")
    
    # Interpretation
    print("\nInterpretation:")
    if avg_bleu > 0.30:
        print("  ⭐⭐⭐ Excellent quality captions!")
    elif avg_bleu > 0.20:
        print("  ⭐⭐ Good quality captions")
    elif avg_bleu > 0.10:
        print("  ⭐ Fair quality captions")
    else:
        print("  Needs improvement - train longer or with more data")
    
    return avg_bleu


def main():
    """Main evaluation function."""
    
    print("="*70)
    print("Image Captioning Evaluation")
    print("="*70)
    
    # Device
    device = config.DEVICE
    print(f"\n✅ Device: {device}")
    
    # ========== Load Vocabulary ==========
    print("\n" + "="*70)
    print("Vocabulary")
    print("="*70)
    
    vocab = load_vocab(config.VOCAB_PATH)
    print(f"✅ Vocabulary size: {len(vocab)}")
    
    # ========== Load Dataset ==========
    print("\n" + "="*70)
    print("Dataset")
    print("="*70)
    
    from dataset import COCODataset
    
    transform = get_transform(image_size=config.IMAGE_SIZE, is_train=False)
    
    dataset = COCODataset(
        root=config.VAL_IMAGES,
        annotation_file=config.VAL_ANNOTATIONS,
        vocab=vocab,
        transform=transform,
        max_length=config.MAX_CAPTION_LENGTH,
        max_samples=config.NUM_EVAL_SAMPLES
    )
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    # ========== Load Model ==========
    print("\n" + "="*70)
    print("Model")
    print("="*70)
    
    # Create model
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        dropout=config.DROPOUT
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    # ========== Evaluate ==========
    print("\n" + "="*70)
    print("Generating Captions")
    print("="*70)
    
    avg_bleu = evaluate(model, dataset, vocab, device, config.NUM_EVAL_SAMPLES)
    
    # ========== Complete ==========
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"✅ Average BLEU-4: {avg_bleu:.4f}")


if __name__ == '__main__':
    main()
