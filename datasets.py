from PIL import Image
import torch


class Img2ImgDataset(torch.utils.data.Dataset):
    def __init__(self, src_images, tgt_images, transform=None):
        self.src_images = src_images
        self.tgt_images = tgt_images
        self.transform = transform

    def __len__(self):
        return len(self.src_images)

    def __getitem__(self, item):
        src_image = Image.open(self.src_images[item])
        tgt_image = Image.open(self.tgt_images[item])

        if self.transform:
            src_image = self.transform(src_image)
            tgt_image = self.transform(tgt_image)

        return src_image, tgt_image
