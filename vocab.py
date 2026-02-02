"""
vocab.py - Vocabulary Builder

Handles word-to-index mapping and vocabulary creation.
"""

import pickle
from collections import Counter
from pycocotools.coco import COCO


class Vocabulary:
    """
    Vocabulary for word ↔ index mapping.
    
    Special tokens:
    - <PAD> (index 0): Padding
    - <START> (index 1): Start of caption
    - <END> (index 2): End of caption
    - <UNK> (index 3): Unknown word
    """
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
        # Add special tokens
        for token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
            self.add_word(token)
    
    def add_word(self, word):
        """Add word to vocabulary."""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __len__(self):
        return len(self.word2idx)
    
    def __call__(self, word):
        """Convert word to index."""
        return self.word2idx.get(word, self.word2idx[self.unk_token])


def build_vocab(annotation_file, threshold=3, max_samples=None):
    """
    Build vocabulary from COCO annotations.
    
    Args:
        annotation_file: Path to COCO annotations JSON
        threshold: Minimum word frequency to include
        max_samples: Limit number of captions (for speed)
    
    Returns:
        Vocabulary object
    """
    print(f"Building vocabulary from {annotation_file}")
    
    coco = COCO(annotation_file)
    counter = Counter()
    ids = list(coco.anns.keys())
    
    # Limit samples if specified
    if max_samples and max_samples < len(ids):
        ids = ids[:max_samples]
        print(f"Using {max_samples} captions (out of {len(coco.anns.keys())})")
    
    # Count words
    for i, ann_id in enumerate(ids):
        caption = str(coco.anns[ann_id]['caption'])
        tokens = caption.lower().split()
        counter.update(tokens)
        
        if (i + 1) % 10000 == 0:
            print(f"Tokenized {i + 1}/{len(ids)} captions")
    
    # Create vocabulary
    vocab = Vocabulary()
    
    # Add words that meet threshold
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)
    
    print(f"✅ Vocabulary size: {len(vocab)} (threshold={threshold})")
    return vocab


def save_vocab(vocab, filepath):
    """Save vocabulary to file."""
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"✅ Vocabulary saved to {filepath}")


def load_vocab(filepath):
    """Load vocabulary from file."""
    with open(filepath, 'rb') as f:
        vocab = pickle.load(f)
    print(f"✅ Vocabulary loaded from {filepath}")
    return vocab


if __name__ == '__main__':
    # Test vocabulary building
    vocab = build_vocab(
        'data/coco/annotations/captions_val2017.json',
        threshold=3,
        max_samples=5000
    )
    
    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Special tokens: {vocab.pad_token}, {vocab.start_token}, {vocab.end_token}, {vocab.unk_token}")
    print(f"PAD index: {vocab.word2idx[vocab.pad_token]}")
    print(f"START index: {vocab.word2idx[vocab.start_token]}")
    print(f"END index: {vocab.word2idx[vocab.end_token]}")
