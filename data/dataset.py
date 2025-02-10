import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', image_size=160):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = 0  # 根据你的数据设置标签
        return image, label
