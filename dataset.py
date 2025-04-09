from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    """ dataset process """

    def __init__(self, images_paths: list, images_names: list, JRD_info_dict: list, transform=None):
        self.images_paths = images_paths
        self.images_names = images_names
        self.JRD_info_dict = JRD_info_dict
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        path = self.images_paths[item]
        img = Image.open(path)
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_paths[item]))
        name = self.images_names[item]
        JRD_label = self.JRD_info_dict[name]
        if self.transform is not None:
            img = self.transform(img)
        return img, name, JRD_label

    @staticmethod
    def collate_fn(batch):
        images, names, JRD_labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        JRD_labels = torch.as_tensor(JRD_labels)
        return images, names, JRD_labels