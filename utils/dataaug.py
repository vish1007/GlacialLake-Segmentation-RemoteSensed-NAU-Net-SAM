#Custom transformation class to apply the same transformation to both images and masks
class JointTransform:
    def __init__(self, prob=0.5, degrees=15):
        self.prob = prob
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Random Horizontal Flip
        if random.random() < self.prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random Vertical Flip
        if random.random() < self.prob:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # Random Rotation
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)

        return {'image': image, 'mask': mask}

# Apply the joint transformations using the custom transform class
train_transforms = JointTransform(prob=0.5, degrees=15)

# Dataset wrapper to apply transformations
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, mask = self.original_dataset[idx]
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['mask']
