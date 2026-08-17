import random
from collections import defaultdict
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch import Tensor
from torch.utils.data import Dataset

from cdcn_msu_mfsd.data.preprocessing import FrameRecord, crop_box, select_uniform


class TargetProvider(Protocol):
    def __call__(self, record: FrameRecord) -> Tensor: ...


class Sample(TypedDict):
    image: Tensor
    target: Tensor
    label: Tensor
    video_id: str
    frame_index: int
    attack_type: str
    device: str


class MSUMFSDDataset(Dataset[Sample]):
    def __init__(
        self,
        records: list[FrameRecord],
        cache_root: Path,
        target_provider: TargetProvider,
        subjects: set[str],
        training: bool,
        frames_per_video: int = 8,
        seed: int = 2026,
        crop_scale: float = 1.3,
        crop_scale_range: tuple[float, float] = (1.2, 1.5),
    ) -> None:
        grouped: dict[str, list[FrameRecord]] = defaultdict(list)
        for record in records:
            if record.subject in subjects:
                grouped[record.video_id].append(record)
        self.records_by_video = {
            key: sorted(value, key=lambda x: x.frame_index) for key, value in grouped.items()
        }
        self.video_ids = sorted(self.records_by_video)
        self.cache_root = cache_root
        self.target_provider = target_provider
        self.training = training
        self.frames_per_video = frames_per_video
        self.seed = seed
        self.crop_scale = crop_scale
        self.crop_scale_range = crop_scale_range
        self.epoch = 0
        if not self.video_ids:
            raise ValueError("Dataset selection contains no videos")

    def __len__(self) -> int:
        return len(self.video_ids) * self.frames_per_video

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _records(self, video_id: str) -> list[FrameRecord]:
        records = self.records_by_video[video_id]
        if not self.training:
            return select_uniform(records, self.frames_per_video)
        generator = random.Random(f"{self.seed}:{self.epoch}:{video_id}")
        if len(records) >= self.frames_per_video:
            return generator.sample(records, self.frames_per_video)
        return [generator.choice(records) for _ in range(self.frames_per_video)]

    def __getitem__(self, index: int) -> Sample:
        video_position, frame_position = divmod(index, self.frames_per_video)
        video_id = self.video_ids[video_position]
        record = self._records(video_id)[frame_position]
        generator = random.Random(f"{self.seed}:{self.epoch}:{video_id}:{record.frame_index}")
        with Image.open(self.cache_root / record.frame_path) as source:
            image = source.convert("RGB")
        scale = generator.uniform(*self.crop_scale_range) if self.training else self.crop_scale
        image = image.crop(crop_box(record, image.width, image.height, scale)).resize((256, 256))
        target = self.target_provider(record)
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(generator.uniform(0.8, 1.2))
            image = ImageEnhance.Contrast(image).enhance(generator.uniform(0.8, 1.2))
            if generator.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                target = target.flip(-1)
        image_array = np.asarray(image, dtype=np.float32).copy()
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - 127.5) / 128.0
        return {
            "image": image_tensor,
            "target": target,
            "label": torch.tensor(record.label, dtype=torch.long),
            "video_id": record.video_id,
            "frame_index": record.frame_index,
            "attack_type": record.attack_type or "genuine",
            "device": record.device,
        }
