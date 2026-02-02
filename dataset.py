"""
dataset.py - COCO Dataset & DataLoader

Handles image loading, caption processing, and batching.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
import torchvision.transforms as transforms


class COCODataset(Dataset):
    """
    COCO Dataset for image captioning.
    
    Returns:
        image: Tensor (3, 224, 224)
        caption: Tensor (max_length,) - token indices
        length: int - actual caption length
    """
    
    def __init__(self, root, annotation_file, vocab, transform=None, 
                 max_length=50, max_samples=None):
        """
        Args:
            root: Path to images directory
            annotation_file: Path to COCO annotations JSON
            vocab: Vocabulary object
            transform: Image transforms
            max_length: Maximum caption length
            max_samples: Limit dataset size (for memory)
        """
        self.root = root
        self.coco = COCO(annotation_file)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length
        
        # Get all annotation IDs
        self.ids = list(self.coco.anns.keys())
        
        # Limit dataset size if specified
        if max_samples and max_samples < len(self.ids):
            self.ids = self.ids[:max_samples]
            print(f"Dataset limited to {max_samples} samples")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        """
        Get one sample.
        
        Returns:
            image: Tensor (3, 224, 224)
            caption: Tensor (max_length,) with padding
            length: int - actual length before padding
        """
        ann_id = self.ids[index]
        annotation = self.coco.anns[ann_id]
        
        # Load image
        img_id = annotation['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Process caption
        caption_text = str(annotation['caption']).lower()
        tokens = caption_text.split()
        
        # Convert to indices: <START> + tokens + <END>
        caption = [self.vocab(self.vocab.start_token)]
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_token))
        
        # Store actual length
        length = len(caption)
        
        # Pad to max_length
        if len(caption) < self.max_length:
            caption.extend([self.vocab(self.vocab.pad_token)] * (self.max_length - len(caption)))
        else:
            caption = caption[:self.max_length]
            length = self.max_length
        
        caption = torch.LongTensor(caption)
        
        return image, caption, length


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Handles variable-length captions by padding.
    
    Args:
        batch: List of (image, caption, length) tuples
    
    Returns:
        images: Tensor (batch_size, 3, 224, 224)
        captions: Tensor (batch_size, max_length)
        lengths: List of actual lengths
    """
    # Sort by length (descending)
    batch.sort(key=lambda x: x[2], reverse=True)
    
    images, captions, lengths = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Stack captions (already padded)
    captions = torch.stack(captions, 0)
    
    return images, captions, list(lengths)


def get_transform(image_size=224, is_train=True):
    """
    Get image transforms.
    
    Args:
        image_size: Target image size
        is_train: Whether training (adds augmentation)
    
    Returns:
        torchvision.transforms.Compose
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloader(root, annotation_file, vocab, batch_size=8, 
                   is_train=True, max_samples=None):
    """
    Get DataLoader for COCO dataset.
    
    Args:
        root: Path to images
        annotation_file: Path to annotations
        vocab: Vocabulary object
        batch_size: Batch size
        is_train: Whether training
        max_samples: Limit dataset size
    
    Returns:
        DataLoader
    """
    # Get transform
    transform = get_transform(is_train=is_train)
    
    # Create dataset
    dataset = COCODataset(
        root=root,
        annotation_file=annotation_file,
        vocab=vocab,
        transform=transform,
        max_samples=max_samples
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    from vocab import build_vocab
    
    # Build vocabulary
    vocab = build_vocab(
        'data/coco/annotations/captions_val2017.json',
        threshold=3,
        max_samples=5000
    )
    
    # Create dataset
    dataset = COCODataset(
        root='data/coco/images/val2017',
        annotation_file='data/coco/annotations/captions_val2017.json',
        vocab=vocab,
        transform=get_transform(is_train=True),
        max_samples=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test one sample
    image, caption, length = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Caption shape: {caption.shape}")
    print(f"Caption length: {length}")
    
    # Decode caption
    words = [vocab.idx2word[idx.item()] for idx in caption[:length]]
    print(f"Caption: {' '.join(words)}")
