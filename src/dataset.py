"""Dataset loader for the Ark Guesser project.

The dataset returns a tuple (x, y) where:
 - x is a torch.FloatTensor of shape (2, num_classes) representing counts
   for the two parts (group0, group1).
 - y is an int (0 or 1) indicating which part won.

 - JSON root contains keys: "names" (mapping name->index) and "data" (list).
 - Within each sample, `groups` is a list with up to 2 dicts. Missing groups
   are treated as empty.
 - Group keys are stringified integer indices that map to class positions.
"""

import json
import os

import torch


class BaseArkGuesserDataset(torch.utils.data.Dataset):
    version: str
    names: dict[int, str]
    entries: list[dict]

    @property
    def num_classes(self):
        return len(self.names)

    @property
    def stats_frequency(self):
        if not hasattr(self, "__cached_stats_frequency"):
            self._init_stats()
        return self.__cached_stats_frequency

    @property
    def stats_quantity(self):
        if not hasattr(self, "__cached_stats_quantity"):
            self._init_stats()
        return self.__cached_stats_quantity

    @property
    def stats_avg_quantity(self):
        if not hasattr(self, "__cached_stats_avg_quantity"):
            self._init_stats()
        return self.__cached_stats_avg_quantity

    def _init_stats(self):
        frequency = {}
        quantity = {}

        for entry in self.entries:
            groups = entry["groups"]
            assert isinstance(groups, list) and len(groups) == 2
            for group_i in range(2):
                group = groups[group_i]
                assert isinstance(group, dict) and len(group) > 0
                for k, v in group.items():
                    # k: class index
                    # v: quantity
                    frequency[int(k)] = frequency.get(k, 0) + 1
                    quantity[int(k)] = quantity.get(k, 0) + v

        self.__cached_stats_frequency = {k: frequency[k] for k in sorted(frequency.keys())}
        self.__cached_stats_quantity = {k: quantity[k] for k in sorted(quantity.keys())}
        self.__cached_stats_avg_quantity = {k: quantity[k] / frequency[k] for k in sorted(quantity.keys())}

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        :returns: FloatTensor(2, num_classes), int
        """
        raise NotImplementedError()

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: FloatTensor(batch_size, 2, num_classes), LongTensor(batch_size)
        """
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([int(b[1]) for b in batch], dtype=torch.long)
        return xs, ys


class RawArkGuesserDataset(BaseArkGuesserDataset):
    def __init__(self, json_path: str):
        super().__init__()
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Dataset json not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.version = raw["version"]
        self.names = raw["names"]
        self.entries = raw["data"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        assert isinstance(entry, dict)
        groups = entry["groups"]
        assert isinstance(groups, list) and len(groups) == 2

        x = torch.zeros((2, self.num_classes), dtype=torch.float32)

        for group_i in range(2):
            group = groups[group_i]
            assert isinstance(group, dict) and len(group) > 0
            for k, v in group.items():
                # k: class index
                # v: quantity
                x[group_i, int(k)] = float(v)

        y = int(entry["winner"])

        return x, y


class RawEvalArkGuesserDataset(RawArkGuesserDataset):
    def __init__(self, json_path: str):
        super().__init__(json_path)

    def get_human_performance(self, idx: int) -> tuple[int, int, int]:
        """
        :returns: correct, wrong, neutral
        """
        entry = self.entries[idx]
        assert isinstance(entry, dict)
        return entry["human_correct"], entry["human_wrong"], entry["human_neutral"]

    def get_game_ground(self, idx: int) -> int:
        entry = self.entries[idx]
        assert isinstance(entry, dict)
        return entry["game_round"]


class AugArkGuesserDataset(BaseArkGuesserDataset):
    SCALE = 4

    def __init__(self, raw_dataset: BaseArkGuesserDataset, subset: torch.utils.data.Subset | None = None):
        self.version = raw_dataset.version
        self.names = raw_dataset.names
        self.entries = raw_dataset.entries
        self._avg_quantity = raw_dataset.stats_avg_quantity
        self._dataset = subset if subset is not None else raw_dataset

    def __len__(self):
        return len(self._dataset) * self.SCALE

    def __getitem__(self, idx: int, advanced: bool = False):
        raw_idx = idx // self.SCALE
        x_raw, y_raw = self._dataset[raw_idx]  # type: ignore
        mod = idx % self.SCALE

        if not advanced:
            if mod == 0:
                # Raw
                return x_raw, y_raw
            elif mod == 1:
                # Swap
                x_flip, y_flip = x_raw[[1, 0], :], 1 - y_raw
                return x_flip, y_flip
            elif mod == 2:
                x_aug = x_raw.clone()
                classes = torch.nonzero(x_aug[y_raw, :] > 0).flatten()
                for cls in classes:
                    x_aug[y_raw, cls] += 1 + self._avg_quantity[int(cls)] // 8
                return x_aug, y_raw
            elif mod == 3:
                x_flip, y_flip = x_raw[[1, 0], :], 1 - y_raw
                x_aug = x_flip.clone()
                classes = torch.nonzero(x_aug[y_flip, :] > 0).flatten()
                for cls in classes:
                    x_aug[y_flip, cls] += 1 + self._avg_quantity[int(cls)] // 8
                return x_aug, y_flip
