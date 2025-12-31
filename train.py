"""
This file is used to train a vision transformer on the CocoDoom dataset.

CocoDoom: https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from Vision.datasets import *
from Vision.Trainer import Trainer
import pycocotools
from pycocotools.coco import COCO


def detr_collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return pixel_values, labels

def create_datasets(processor):
    # Load COCO annotations
    coco_doom_dataset_path = os.path.join(
        os.pardir, "datasets", "cocodoom"
    )
    
    # we're using the run split,
    # so run1 = train, run2 = val, and run3 = test
    train_annotation_file = "run-train.json"
    val_annotation_file = "run-val.json"
    test_annotation_file = "run-test.json"

    train_dataset = CocoDoomDataset(
        data_dir=coco_doom_dataset_path,
        annotation_file_name=train_annotation_file,
        processor=processor
    )
    val_dataset = CocoDoomDataset(
        data_dir=coco_doom_dataset_path,
        annotation_file_name=val_annotation_file,
        processor=processor
    )
    # test_dataset = CocoDoomDataset(
    #     data_dir=coco_doom_dataset_path,
    #     annotation_file_name=test_annotation_file,
    #     processor=processor
    # )

    return train_dataset, val_dataset


def main():
    optimizer_lr = 1e-5
    weight_decay = 1e-4
    batch_size = 8
    num_epochs = 10
    workers = 16
    
    # Training setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50")

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(processor)

    lru_cache_dataset = DiskCachedDataset(
        train_dataset,
        cache_dir=os.path.join(
            os.pardir, "datasets", "cocodoom", "preprocessed"
        ),
        max_cache_items=5000
    )

    train_dataloader = DataLoader(
        lru_cache_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=detr_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=detr_collate_fn
    )

    # Load the model
    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        ignore_mismatched_sizes=True,
        num_labels=len(train_dataset.coco.getCatIds()),
        id2label={i: cat['name'] for i, cat in enumerate(train_dataset.id2label)},
        label2id={cat['name']: i for i, cat in enumerate(train_dataset.id2label)}
    ).to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=weight_decay
    )
    # Create Trainer
    trainer = Trainer()
    trainer.train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        num_epochs=num_epochs
    )

    print("\nTraining complete!")

if __name__ == "__main__":
    main()