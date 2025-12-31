# Add project directory to path for imports
import torch
from transformers import DetrImageProcessor
from datasets import CocoDoomDataset
from PIL import Image
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.pardir))



# create preprocessor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# create dataset instance
train_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-train.json",
    processor=processor
)
val_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-val.json",
    processor=processor
)
test_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-test.json",
    processor=processor
)


def preprocess_and_save_dataset(dataset, split_name):
    saved, skipped = 0, 0
    save_root = os.path.join(
        os.pardir, os.pardir, "datasets", "cocodoom", "preprocessed"
    )
    os.makedirs(save_root, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc=f"Preprocessing {split_name}"):
        image, target, img_file_name = dataset.get_image(i)

        encoding = processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding['pixel_values'].squeeze()
        target = dict(encoding['labels'][0])

        # reduce format of target tensors
        target['boxes'] = target['boxes'].to(torch.float16)
        target['size'] = None
        # we only have 94 categories
        target['class_labels'] = target['class_labels'].to(torch.int16)
        target['area'] = None  # remove area to save space
        target['iscrowd'] = None  # remove iscrowd to save space

        # modify file name to have .pt extension
        pt_file_name = os.path.splitext(img_file_name)[0] + ".pt"
        save_path = os.path.join(save_root, pt_file_name)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            skipped += 1
            continue

        torch.save(
            {
                "pixel_values": pixel_values,
                "labels": target
            },
            save_path
        )
        saved += 1

    print(f"{split_name}: saved {saved}, skipped {skipped}")


# Preprocess and save datasets
print("Preprocessing training dataset...")
preprocess_and_save_dataset(train_dataset, "train")
print("Preprocessing validation dataset...")
preprocess_and_save_dataset(val_dataset, "val")
# preprocess_and_save_dataset(test_dataset, "test")
