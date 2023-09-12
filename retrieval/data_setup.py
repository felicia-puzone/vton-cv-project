import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pathlib
from PIL import Image
import math
import pickle
import copyreg

import config_file as cfg


class CustomDataset(Dataset):
    def __init__(self, target_dir: str, target_file: str, transform=None) -> None:
        self.target_dir = pathlib.Path(target_dir)
        self.target_in_shop_dir = pathlib.Path(target_dir) / "in-shop-cloth"
        self.target_worn_dir = pathlib.Path(target_dir) / "worn-cloth"
        self.target_file = pathlib.Path(target_file)
        self.cloth_names_list = self._load_target_data(self.target_file)

        self.transform = transform

        self.module_action = cfg.N_NEGATIVE_EXAMPLES * 2
        self.half_module_action = cfg.N_NEGATIVE_EXAMPLES
        self.real_len = len(self.cloth_names_list)
        self.virtual_len = self.real_len * self.module_action

    @staticmethod
    def _load_target_data(target_file):
        names_list = []
        with open(target_file, 'r') as file:
            for line in file:
                name = line.split(sep=".")[0]
                names_list.append(name)

        return names_list

    def __len__(self) -> int:
        return self.virtual_len

    def _prepare_tensor(self, index: int, is_worn: bool = True):
        name = self.cloth_names_list[index]
        if is_worn:
            features_path = self.target_worn_dir / (name + ".pickle")
        else:
            features_path = self.target_in_shop_dir / (name + ".pickle")

        features = None
        try:
            with open(features_path.as_posix(), 'rb') as features_file:
                features = pickle.load(features_file)
        except pickle.PickleError:
            print(f"Something went wrong in the loading of {self.cloth_names_list[index]} data!!!")
        if features is None:
            raise ValueError(f"From {self.cloth_names_list[index]} no loaded data!!!")

        tensor = self.transform(features)

        return tensor

    def __getitem__(self, index: int):
        real_index = math.floor(index / self.module_action)
        action = index % self.module_action

        if action < self.half_module_action:
            label = 1
            in_shop_tensor = self._prepare_tensor(real_index, is_worn=False)
            worn_tensor = self._prepare_tensor(real_index, is_worn=True)
            if action != 0:
                worn_tensor += torch.normal(mean=0.0, std=0.005, size=worn_tensor.shape)
        else:
            label = 0
            rand_index = int(torch.randint(low=0, high=self.real_len, size=(1,)))
            in_shop_tensor = self._prepare_tensor(real_index, is_worn=False)
            worn_tensor = self._prepare_tensor(rand_index, is_worn=True)

        in_shop_tensor = torch.flatten(in_shop_tensor)
        worn_tensor = torch.flatten(worn_tensor)
        tensor = torch.cat((in_shop_tensor, worn_tensor), dim=-1)

        return tensor, label


def create_dataloader(target_dir: str,
                      train_file: str, test_file: str,
                      transform: transforms.Compose,
                      batch_size: int = cfg.BATCH_SIZE,
                      num_workers: int = cfg.NUM_WORKERS):
    train_data = CustomDataset(target_dir, train_file, transform=transform)
    test_data = CustomDataset(target_dir, test_file, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=cfg.PIN_MEMORY)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 pin_memory=cfg.PIN_MEMORY)

    return train_dataloader, test_dataloader, cfg.CLASS_NAMES
