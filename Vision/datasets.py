import os
import torch
from torch.utils.data import Dataset
import torchvision
from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
from collections import OrderedDict
import threading

class CocoDoomDataset(torchvision.datasets.CocoDetection):
    """
    Custom COCO dataset for training a DETR model.
    """
    def __init__(
        self, data_dir, annotation_file_name, 
        processor
    ):
        """
        Args:
            data_dir: Path to dataset
            annotation_file_name: Name of the COCO annotation file
            processor: DETR image processor
        """
        # load COCO annotation
        annotation_file = os.path.join(
            data_dir, annotation_file_name)
        super().__init__(root=data_dir, annFile=annotation_file)

        self.img_folder = data_dir
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.id2label = self.coco.loadCats(self.coco.getCatIds())
        # Map COCO category ids (which may be non-contiguous) to contiguous 0-based ids
        # expected by DETR's classification head.
        self.cat_id_to_contiguous_id = {
            cat_id: idx for idx, cat_id in enumerate(self.coco.getCatIds())
        }
        self.ids = list(sorted(self.coco.imgs.keys()))

        print(f"Loaded {annotation_file_name}")
        print(f"Number of images: {len(self.coco.imgs)}")
        print(f"Number of Categories: {len(self.coco.getCatIds())}")

    def __len__(self):
        return len(self.ids)

    # def get_preprocessed_item(self, idx):
    def fast__getitem__(self, idx):
        """
        Get a single preprocessed image and target.
        
        Uses preprocessed data from disk.
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(
            self.img_folder, "preprocessed", img_info['file_name'])
        img_path = img_path.replace('.png', '.pt')
                
        data = torch.load(img_path, weights_only=True)
        pixel_values = data['pixel_values']
        target = data['labels']
        
        return pixel_values, target

    def __getitem__(self, idx):
        """
        Get a single image and its target.

        Uses the processor.
        """
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        # Remap category ids to contiguous range [0, num_classes-1]
        remapped_annotations = []
        for ann in target:
            ann_copy = ann.copy()
            ann_copy['category_id'] = self.cat_id_to_contiguous_id[ann['category_id']]
            remapped_annotations.append(ann_copy)

        target = {'image_id': img_id, 'annotations': remapped_annotations}

        # preprocess data
        encoding = self.processor(
            images=img,
            annotations=target,
            return_tensors="pt"
        )
        
        # squeeze and [0] to remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


    def get_image(self, idx):
        """
        Get a single image by index.
        """
        # For DETR models, we can use this line of code to
        # obtain appropriate structure for the annotations
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        target = {'image_id': img_id, 'annotations': target}
        return img, target, img_info["file_name"]


class LRUCachedDataset(Dataset):
    def __init__(self, dataset, max_cache_items=100):
        self.dataset = dataset
        self.cache = OrderedDict()
        self.max_cache_items = max_cache_items

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(idx)
            return self.cache[idx]

        item = self.dataset[idx]

        # Add to cache
        self.cache[idx] = item
        if len(self.cache) > self.max_cache_items:
            # Remove oldest item
            self.cache.popitem(last=False)

        return item


class DiskCachedDataset(Dataset):
    def __init__(self, dataset, cache_dir, max_cache_items=100):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_cache_items = max_cache_items
        self.cached_items = 0
        self._cache_lock = threading.Lock()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        cache_file = self.cache_dir / f"{idx}.pt"

        if cache_file.exists():
            item = torch.load(cache_file, weights_only=True)
            return item['pixel_values'], item['labels']

        pixel_values, labels = self.dataset[idx]
        self._cache_item(idx, pixel_values, labels)
        return pixel_values, labels

    def _cache_item(self, idx, pixel_values, labels):
        with self._cache_lock:
            cache_file = self.cache_dir / f"{idx}.pt"

            if not cache_file.exists():
                data_to_cache = {
                    'pixel_values': pixel_values,
                    'labels': labels
                }
                torch.save(data_to_cache, cache_file)
                self.cached_items += 1
            
            if self.cached_items > self.max_cache_items:
                # Remove first item
                first_cache_file = os.listdir(self.cache_dir)[0]
                os.remove(self.cache_dir / first_cache_file)
                self.cached_items -= 1