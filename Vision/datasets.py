import os
import torch
from torch.utils.data import Dataset
import torchvision
from pycocotools.coco import COCO
from pathlib import Path
from collections import OrderedDict


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
        cat_list = self.coco.loadCats(self.coco.getCatIds())

        # create contiguous category id mapping to handle
        # inconsistent category ids in CocoDoom
        sorted_cats = sorted(cat_list, key=lambda x: x['id'])
        self.coco_id_to_contiguous = {
            cat['id']: idx for idx, cat in enumerate(sorted_cats)
        }
        # remap ids to contiguous ids
        self.id2cat = {
            idx: cat['name'] for idx, cat 
            in enumerate(sorted_cats)
        }
        self.cat2id = {
            cat['name']: idx for idx, cat
            in enumerate(sorted_cats)
        }
        self.num_categories = len(self.id2cat)

        # identifier for each image in dataset
        self.img_ids = list(self.coco.imgs.keys())      

        print(f"Loaded {annotation_file_name}")
        print(f"Number of images: {len(self.coco.imgs)}")
        print(f"Number of Categories: {len(self.coco.getCatIds())}")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        Get a single image and its target.

        Uses the processor.
        """
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.img_ids[idx]

        # remap category IDs from image to consistent format
        for ann in target:
            original_id = ann['category_id']
            ann['category_id'] = self.coco_id_to_contiguous[original_id]

        target = {'image_id': img_id, 'annotations': target}

        # preprocess data
        encoding = self.processor(
            images=img,
            annotations=target,
            return_tensors="pt"
        )
        
        # squeeze and [0] to remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = encoding["labels"][0]

        return pixel_values, target


    def get_image(self, idx):
        """
        Get a single image by index.
        """
        # For DETR models, we can use this line of code to
        # obtain appropriate structure for the annotations
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        target = {'image_id': img_id, 'annotations': target}
        return img, target, img_info["file_name"]


    def get_img_info(self, idx):
        """
        Returns the image ID and path
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(
            self.img_folder, "preprocessed", img_info['file_name'])

        return img_id, img_path


class CachedDataset(Dataset):
    def __init__(self, dataset, cache_size_gb=8):
        self.dataset = dataset
        self.cache = {}

        # use uniform distribution to sample what idx should be cached
        dist = torch.distributions.Uniform(0, len(dataset))
        approx_item_size = 0.012 # in GB
        num_items_to_cache = int(cache_size_gb / approx_item_size)
        sampled_indices = dist.sample((num_items_to_cache,)).long().tolist()

        for idx in sampled_indices:
            self.cache[idx] = dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.dataset[idx]
        return item


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


class PreprocessedDataset(Dataset):
    """
    This dataset is a wrapper around CocoDoomDataset that
    returns preprocessed images from disk, if they exist.
    """
    def __init__(self, dataset, cache_dir):
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_id, img_path = self.dataset.get_img_info(idx)
        cache_file = self.cache_dir / (Path(img_path).stem + ".pt")

        if cache_file.exists():
            item = torch.load(cache_file, weights_only=True)
            return item['pixel_values'], item['labels']

        pixel_values, labels = self.dataset[idx]
        return pixel_values, labels