import json
import os

from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, cfg, data_path):
        super().__init__()
        self.cfg = cfg
        with open(data_path, "r") as fp:
            data_list = list(fp)

        self.data = []
        for sample in data_list:
            sample = json.loads(sample)

            if "context" in sample and isinstance(sample["context"], list):
                sample["context"] = " ".join(sample["context"])

            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_data(cfg):
    train_data = SimpleDataset(cfg, os.path.join(cfg.data, cfg.train_file))
    val_data = SimpleDataset(cfg, os.path.join(cfg.data, cfg.val_file))
    if os.path.exists(os.path.join(cfg.data, cfg.test_file)):
        test_data = SimpleDataset(cfg, os.path.join(cfg.data, cfg.test_file))
    else:
        test_data = None
    return train_data, val_data, test_data
