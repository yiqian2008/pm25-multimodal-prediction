from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HoornImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Tabular features
        tabular = torch.tensor(
            [float(row["population"])],
            dtype=torch.float32
        )

        # Target label
        label = torch.tensor(float(row["pm25"]), dtype=torch.float32)

        return image, tabular, label