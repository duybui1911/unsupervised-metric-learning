import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


class ShoesDataset(Dataset):
      def __init__(self, folder_paths, folder_labels, transform=None):
            self.data = folder_paths
            self.transform = transform
            self.labels = folder_labels

      def __len__(self):
            return len(self.data)

      def __getitem__(self, idx):
            label = self.labels[idx]
            try:
                img1 = Image.open(osp.join(self.data[idx])).convert('RGB')
            except:
                print(self.data[idx])
                
            if self.transform:
                img1 = self.transform(img1)

            return tuple(img1["global_crops"] + img1["local_crops"] + [label])
