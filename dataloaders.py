from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_dataloaders(batch_size=64, image_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CelebA(
        root='data',
        split='train',
        transform=transform,
        download=True
    )

    val_dataset = datasets.CelebA(
        root='data',
        split='valid',
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader