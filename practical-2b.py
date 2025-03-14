from pathlib import Path

from PIL import Image
from torchvision.datasets import VisionDataset


class ColorGrayImageDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(Path(root), transform, target_transform)

        self.files = [self.root/file for file in self.root.iterdir() if file.suffix in (".jpg", ".png")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        label = (len(image.mode)-1) // 2

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


dataset = ColorGrayImageDataset("data/")
