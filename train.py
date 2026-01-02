"""
This file is used to train a vision transformer on the CocoDoom dataset.

CocoDoom: https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
# facebook/detr
from transformers import DetrImageProcessorFast, AutoModelForObjectDetection
# rt-detr
from transformers import RTDetrForObjectDetection, RTDetrImageProcessorFast
from Vision.datasets import *
from Vision.Trainer import Trainer
from pycocotools.coco import COCO
from ultralytics import YOLO


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


def train_detr_model(device):
    optimizer_lr = 1e-3
    weight_decay = 1e-4
    batch_size = 16
    num_epochs = 10
    workers = 16
    # model_str = "PekingU/rtdetr_r18vd"
    model_str = "facebook/detr-resnet-50"

    # Load processor
    processor = DetrImageProcessorFast.from_pretrained(
        model_str,
        size={"shortest_edge": 200, "longest_edge": 320},
        # do_resize=True,
        # size={"max_height": 640, "max_width": 640},
        # do_pad=True,
        # pad_size={"height": 640, "width": 640},
        use_fast=True
    )

    # Create datasets and dataloaders
    train_dataset, val_dataset = create_datasets(processor)

    train_dataloader = DataLoader(
        train_dataset,
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
        model_str,
        ignore_mismatched_sizes=True,
        num_labels=train_dataset.num_categories,
        id2label=train_dataset.id2cat,
        label2id=train_dataset.cat2id
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



def train_yolo_model(device):
    # load a pretrained model
    model = YOLO("yolo11n.pt")
    yaml_path = os.path.join(
        os.pardir, "datasets", "cocodoom",
        "yolo", "data.yaml")

    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=384,
        batch=16,
        device=device
    )
    # print(results)
    print("\nYOLO Training complete!")


def main():
    # Training setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    train_detr_model(device)
    # train_yolo_model(device)

    

if __name__ == "__main__":
    main()