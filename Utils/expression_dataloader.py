import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images_per_class=30):
        self.root_dir = root_dir
        self.transform = transform
        self.max_images_per_class = max_images_per_class
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()

    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            img_names = os.listdir(class_dir)
            img_names = img_names[:self.max_images_per_class]
            for img_name in img_names:
                img_path = os.path.join(class_dir, img_name)
                images.append((img_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_data_loader(root_dir, batch_size=100, max_images_per_class=30, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = FacialExpressionDataset(root_dir=root_dir, transform=transform, max_images_per_class=max_images_per_class)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return data_loader