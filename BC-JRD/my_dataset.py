from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet_d(Dataset):
    """自定义数据集"""

    def __init__(self, images_paths: list, images_labels: list, images_names: list, transform=None, normalization=None):
        self.images_paths = images_paths
        self.images_labels = images_labels
        self.images_names = images_names
        self.transform = transform
        self.normalization = normalization

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        #path = self.images_paths[item][15:]
        path = self.images_paths[item]
        img = Image.open(path)
        # print(type(img))  输出: <class 'PIL.PngImagePlugin.PngImageFile'>
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_paths[item]))
        img = self.transform(img)
        img = self.normalization(img)
        label = self.images_labels[item]
        name = self.images_names[item]
        return img, label, name

    @staticmethod
    def collate_fn(batch):
        images, labels, names = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, names