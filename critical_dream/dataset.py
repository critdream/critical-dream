import itertools
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms
from torchvision.transforms.functional import crop
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


@dataclass
class InstanceConfig:
    instance_name: str
    instance_data_root: str
    instance_prompt: str
    instance_urls: list[str]
    class_data_root: str | None = None
    class_prompt: str | None = None
    validation_prompt: str | None = None


@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_config_name: str
    image_column: str | None = None
    caption_column: str | None = None
    resolution: int = 256
    random_flip: bool = True
    center_crop: bool = False


class DreamBoothMultiInstanceDataset(Dataset):
    """
    A dataset that composes multiple DreamBoothDataset objects so that you
    can train multiple instances in one go.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        multi_instance_data_config: list[InstanceConfig],
        multi_instance_subset: list[str] | None = None,
        data_dir_root: Path | None = None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
    ):
        self.custom_instance_prompts = True

        _multi_instance_subset = multi_instance_subset or []

        with open(multi_instance_data_config) as f:
            self.multi_instance_data_config = [
                InstanceConfig(**x) for x in yaml.safe_load(f)
                if (
                    True
                    if not _multi_instance_subset
                    else x["instance_name"] in _multi_instance_subset
                )
            ]

        self.dreambooth_datasets = {
            config.instance_name: DreamBoothDataset(
                dataset_config,
                instance_data_root=(
                    config.instance_data_root
                    if data_dir_root is None
                    else data_dir_root / config.instance_data_root
                ),
                instance_prompt=config.instance_prompt,
                class_prompt=config.class_prompt,
                class_data_root=(
                    config.class_data_root
                    if data_dir_root is None
                    else data_dir_root / config.class_data_root
                ),
                class_num=class_num,
                size=size,
                repeats=repeats,
                center_crop=center_crop,
                custom_instance_prompts=self.custom_instance_prompts,
            )
            for config in self.multi_instance_data_config
        }

        global_index = 0
        self.multi_dataset_index = {}
        self.local_dataset_index = {}
        for instance_name, dataset in self.dreambooth_datasets.items():
            for local_index in range(len(dataset)):
                self.multi_dataset_index[global_index] = instance_name
                self.local_dataset_index[global_index] = local_index
                global_index += 1

        self._length = len(self.multi_dataset_index)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        dataset = self.dreambooth_datasets[self.multi_dataset_index[index]]
        example = dataset[self.local_dataset_index[index]]
        return example


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        instance_data_root,
        instance_prompt,
        class_prompt,
        class_data_root=None,
        class_num=None,
        size=1024,
        repeats=1,
        center_crop=False,
        custom_instance_prompts: bool | list = False,
    ):
        self.size = size
        self.center_crop = center_crop

        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = custom_instance_prompts
        self.class_prompt = class_prompt

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        if dataset_config.dataset_name is not None:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError(
                    "You are trying to load your data using the datasets library. If you wish to train using custom "
                    "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                    "local folder containing images only, specify --instance_data_dir instead."
                )
            # Downloading and loading a dataset from the hub.
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
            dataset = load_dataset(
                dataset_config.dataset_name,
                dataset_config.dataset_config_name,
                cache_dir=dataset_config.cache_dir,
            )
            # Preprocessing the datasets.
            column_names = dataset["train"].column_names

            # 6. Get the column names for input/target.
            if dataset_config.image_column is None:
                image_column = column_names[0]
                logger.info(f"image column defaulting to {image_column}")
            else:
                image_column = dataset_config.image_column
                if image_column not in column_names:
                    raise ValueError(
                        f"`--image_column` value '{dataset_config.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
            instance_images = dataset["train"][image_column]

            if dataset_config.caption_column is None:
                logger.info(
                    "No caption column provided, defaulting to instance_prompt for all images. If your dataset "
                    "contains captions/prompts for the images, make sure to specify the "
                    "column as --caption_column"
                )
                self.custom_instance_prompts = None
            else:
                if dataset_config.caption_column not in column_names:
                    raise ValueError(
                        f"`--caption_column` value '{dataset_config.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                    )
                custom_instance_prompts = dataset["train"][dataset_config.caption_column]
                # create final list of captions according to --repeats
                self.custom_instance_prompts = []
                for caption in custom_instance_prompts:
                    self.custom_instance_prompts.extend(itertools.repeat(caption, repeats))
        else:
            self.instance_data_root = Path(instance_data_root)
            if not self.instance_data_root.exists():
                raise ValueError(f"Instance images root {self.instance_data_root} doesn't exists.")

            instance_images = [Image.open(path) for path in list(Path(instance_data_root).iterdir())]

        self.instance_images = []
        for img in instance_images:
            self.instance_images.extend(itertools.repeat(img, repeats))

        # image processing to prepare for using SD-XL micro-conditioning
        self.original_sizes = []
        self.crop_top_lefts = []
        self.pixel_values = []
        train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        train_flip = transforms.RandomHorizontalFlip(p=1.0)
        train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            self.original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if dataset_config.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if dataset_config.center_crop:
                y1 = max(0, int(round((image.height - dataset_config.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - dataset_config.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(image, (dataset_config.resolution, dataset_config.resolution))
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            self.crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            self.pixel_values.append(image)

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.pixel_values[index % self.num_instance_images]
        original_size = self.original_sizes[index % self.num_instance_images]
        crop_top_left = self.crop_top_lefts[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["original_size"] = original_size
        example["crop_top_left"] = crop_top_left

        if self.custom_instance_prompts:
            try:
                caption = self.custom_instance_prompts[index % self.num_instance_images]
            except TypeError:
                caption = None
            if caption:
                example["instance_prompt"] = caption
            else:
                example["instance_prompt"] = self.instance_prompt

        else:  # costum prompts were provided, but length does not match size of image dataset
            example["instance_prompt"] = self.instance_prompt

        if self.class_data_root and self.num_class_images > 0:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt"] = self.class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_top_lefts = [example["crop_top_left"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "original_sizes": original_sizes,
        "crop_top_lefts": crop_top_lefts,
    }
    return batch
