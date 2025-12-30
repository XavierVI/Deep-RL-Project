import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class CocoDetection(Dataset):
    """Custom COCO dataset for DETR training."""

    def __init__(self, img_folder, annotation_file, processor, catid2contig, augment=False):
        """
        Args:
            img_folder: Path to image folder
            annotation_file: Path to COCO format JSON annotation file
            processor: DETR image processor for preprocessing
            catid2contig: Mapping from original COCO category ids to contiguous ids
            augment: Whether to apply data augmentation
        """
        self.img_folder = img_folder
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.catid2contig = catid2contig
        self.augment = augment
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image info
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations in COCO format (xywh, absolute pixels)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Filter out crowd / invalid boxes and ensure required keys
        coco_annotations = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("bbox") is None:
                continue
            cat_contig = self.catid2contig[ann["category_id"]]
            coco_annotations.append({
                # [x, y, w, h] in absolute pixels
                "bbox": ann["bbox"],
                "category_id": cat_contig,       # remapped to contiguous ids
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0)
            })

        # Build target in the format expected by the processor
        target = {
            "image_id": img_id,
            "annotations": coco_annotations
        }

        # Process image and target with DETR processor
        encoding = self.processor(
            images=image, annotations=target, return_tensors="pt")

        # Remove batch dimension added by processor
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels
