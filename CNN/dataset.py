from torchvision import datasets, transforms

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

num_workers = 8

data_dir = "../imagenet2012/imagenet"

train_dataset = datasets.ImageFolder(
    root=f"{data_dir}/train",
    transform=transformations
)

test_dataset = datasets.ImageFolder(
    root=f"{data_dir}/test",
    transform=test_transformations
)
'''
train_dataset = datasets.CIFAR100(
    root=data_dir,
    download=False,
    transform=transformations,
)

test_dataset = datasets.CIFAR100(
    root=data_dir,
    download=False,
    transform=transformations,
)
'''