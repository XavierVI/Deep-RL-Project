"""
This file is used to train a vision transformer on the CocoDoom dataset.

CocoDoom: https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AdamW
from Vision.datasets import CocoDetection
from Vision.Trainer import Trainer

def update_label_mappings(coco):
    """Create mappings between original COCO category ids and contiguous ids."""
    cat_ids = coco.getCatIds()
    catid2contig = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    contig2catid = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    id2label = {i: coco.loadCats(cat_id)[0]['name'] for i, cat_id in enumerate(cat_ids)}
    label2id = {v: k for k, v in id2label.items()}
    return catid2contig, contig2catid, id2label, label2id

def main():
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize processor and model
    processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # Update config metadata to match custom dataset
    model.config.num_labels = num_classes
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Update model's classification head if needed
    # DETR uses 91 classes by default (COCO dataset)
    # If your dataset has different number of classes, you need to adjust
    if model.config.num_labels != num_classes:
        print(
            f"Adjusting model from {model.config.num_labels} to {num_classes} classes")
        model.class_labels_classifier = torch.nn.Linear(
            model.config.d_model, num_classes + 1  # +1 for "no object" class
        )
    else:
        # Ensure head matches the updated num_classes when config already set
        model.class_labels_classifier = torch.nn.Linear(
            model.config.d_model, num_classes + 1
        )

    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    # Training parameters
    num_epochs = 10



    print("\nTraining complete!")

if __name__ == "__main__":
    main()