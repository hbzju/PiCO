import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from .randaugment import RandomAugment
from .utils_algo import generate_uniform_cv_candidate_labels,generate_hierarchical_cv_candidate_labels

def load_cifar100(partial_rate, batch_size, hierarchical, noisy_rate=0):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    temp_train = dsets.CIFAR100(root='./data', train=True, download=True)
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
    # get original data and labels

    test_dataset = dsets.CIFAR100(root='./data', train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4,
        sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
    
    if hierarchical:
        partialY = generate_hierarchical_cv_candidate_labels('cifar100', labels, partial_rate, noisy_rate=noisy_rate)
        # for fine-grained classification
    else:
        partialY = generate_uniform_cv_candidate_labels(labels, partial_rate, noisy_rate=noisy_rate)
    
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('Running defualt PLL setting')
    else:
        print('Running noisy PLL setting')
    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = CIFAR100_Augmentention(data, partialY.float(), labels.float())
    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True)
    return partial_matrix_train_loader,partialY,train_sampler,test_loader


class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        return each_image_w, each_image_s, each_label, each_true_label, index

