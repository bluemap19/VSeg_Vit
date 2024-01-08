# config.py
import os
from torchvision import transforms

use_gpu=True
gpu_name=1

pre_model=os.path.join('pth','model.pth')

save_path="pth"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


# loaddataset.py
from torchvision.datasets import CIFAR10
from PIL import Image


class PreDataset(CIFAR10):
    def __getitem__(self, item):
        img, target=self.data[item], self.targets[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            imgL = self.transform(img)
            imgR = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return imgL, imgR, target


if __name__=="__main__":

    import config
    train_data = PreDataset(root='dataset', train=True, transform=config.train_transform, download=True)
    print(train_data[51])
