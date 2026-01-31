# -*- coding: utf-8 -*-
# @Author : Cheng Tan
# @Email  : tancheng@westlake.edu.cn
# @File   : noise_datasets.py

import numpy as np
import torchvision
from PIL import Image
import torch
from datasets.cifar import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from mydataset import MyDataSet
from myload_data import read_datalist, load_per_data

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class NoiseDataset(torchvision.datasets.VisionDataset):
    def __init__(
            self, 
            noise_type: str = 'none',
            asym_trans: dict = None, 
            percent: float = 0.0,
        ) -> None:
        
        assert percent <= 1.0 and percent >= 0.0
        assert noise_type in ['sym', 'asym', 'pair', 'ins', 'ins_dep', 'none']

        self.percent = percent
        self.noise_type = noise_type
        self.asym_trans = asym_trans
        self.images_paths, self.labels = read_datalist()

        # dataset info
        self.min_target = min(self.labels)
        self.max_target = max(self.labels)
        self.num_classes = len(np.unique(self.labels))
        assert self.num_classes == self.max_target - self.min_target + 1
        self.num_samples = len(self.labels)
        
        if self.noise_type == 'sym':
            self.symmetric_noise()
        elif self.noise_type == 'asym':
            self.asymmetric_noise()
        elif self.noise_type == 'pair':
            self.pair_noise()
        elif self.noise_type == 'ins':
            self.instance_noise(tau=self.percent)
        elif self.noise_type == 'ins_dep':
            self.instance_dependent_noise()

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.images_paths))
        for i, idx in enumerate(indices):
            if i < self.percent * len(self.images_paths):
                self.labels[idx] = np.random.randint(low=self.min_target, high=self.max_target + 1, dtype=np.int32)

    def asymmetric_noise(self):
        target_copy = self.labels.copy()
        if self.asym_trans == None:
            indices = np.arange(self.num_samples)
            np.random.shuffle(indices)
            idx = indices[:int(self.percent * self.num_samples)]
            target_copy = np.array(target_copy)
            target_copy[idx] = (target_copy[idx] + 1) % (self.max_target + 1) + self.min_target
            self.labels = target_copy
        else:
            for i in self.asym_trans.keys():
                indices = list(np.where(np.array(target_copy) == i)[0])
                np.random.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j <= self.percent * len(indices):
                        self.labels[idx] = (self.asym_trans[i] if i in self.asym_trans.keys() else i)
        del target_copy

    def pair_noise(self):
        target_copy = self.labels.copy()
        if self.asym_trans is None:
            indices = np.arange(len(self.labels))
            np.random.shuffle(indices)
            idx = indices[:int(self.percent * len(self.labels))]
            target_copy = np.array(target_copy)
            target_copy[idx] = (target_copy[idx] + 1) % 3
            self.labels = target_copy
        else:
            for i in range(3):
                indices = list(np.where(np.array(target_copy) == i)[0])
                np.random.shuffle(indices)
                for j, idx in enumerate(indices):
                    if j < self.percent * len(indices):
                        self.labels[idx] = (i + 1) % 3
        del target_copy


    def instance_noise(
            self,
            tau: float = 0.2, 
            std: float = 0.1, 
            feature_size: int = 3 * 32 * 32, 
            # seed: int = 1
        ): 
        '''
        Thanks the code from https://github.com/SML-Group/Label-Noise-Learning wrote by SML-Group.
        LabNoise referred much about the generation of instance-dependent label noise from this repo.
        '''
        from scipy import stats
        from math import inf
        import torch.nn.functional as F

        # np.random.seed(int(seed))
        # torch.manual_seed(int(seed))
        # torch.cuda.manual_seed(int(seed))
                
        # common-used parameters
        num_samples = self.num_samples
        num_classes = self.num_classes

        P = []
        # sample instance flip rates q from the truncated normal distribution N(\tau, {0.1}^2, [0, 1])
        flip_distribution = stats.truncnorm((0 - tau) / std, (1 - tau) / std, loc=tau, scale=std)
        '''
        The standard form of this distribution is a standard normal truncated to the range [a, b]
        notice that a and b are defined over the domain of the standard normal. 
        To convert clip values for a specific mean and standard deviation, use:

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        truncnorm takes  and  as shape parameters.

        so the above `flip_distribution' give a truncated standard normal distribution with mean = `tau`,
        range = [0, 1], std = `std`
        '''
        # how many random variates you need to get
        q = flip_distribution.rvs(num_samples)
        # sample W \in \mathcal{R}^{S \times K} from the standard normal distribution N(0, 1^2)
        W = torch.tensor(np.random.randn(num_classes, feature_size, num_classes)).float().to(device)
        for i in range(num_samples):
            x, y = self.transform(Image.fromarray(self.images_paths[i])), torch.tensor(self.labels[i])
            x = x.to(device)
            # step (4). generate instance-dependent flip rates
            # 1 x feature_size  *  feature_size x 10 = 1 x 10, p is a 1 x 10 vector
            p = x.reshape(1, -1).mm(W[y]).squeeze(0)
            # step (5). control the diagonal entry of the instance-dependent transition matrix
            # As exp^{-inf} = 0, p_{y} will be 0 after softmax function.
            p[y] = -inf 
            # step (6). make the sum of the off-diagonal entries of the y_i-th row to be q_i
            p = q[i] * F.softmax(p, dim=0)
            p[y] += 1 - q[i]
            P.append(p)
        P = torch.stack(P, 0).cpu().numpy()
        l = [i for i in range(self.min_target, self.max_target + 1)]
        new_label = [np.random.choice(l, p=P[i]) for i in range(num_samples)]

        print('noise rate = ', (new_label != np.array(self.labels)).mean())
        self.labels = new_label

    def instance_dependent_noise(self):
        import torch
        import random
        from torch import nn
        
        # 简单的特征提取器（如果没有提供）
        class SimpleFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
            
            def forward(self, x):
                return self.flatten(x)
        
        feature_extractor = SimpleFeatureExtractor().to(device)
        
        # 加载和预处理图像
        images = []
        labels = torch.tensor(self.labels).to(device)
        
        for img_path in self.images_paths:
            img = load_per_data(img_path)
            img_array = np.array(img)
            img = Image.fromarray(img_array)
            if hasattr(self, 'transform') and self.transform is not None:
                img = self.transform(img)
            else:
                # 默认转换
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                img = transform(img)
            images.append(img)
        
        images = torch.stack(images).to(device)
        
        with torch.no_grad():
            # 1. 提取特征
            features = feature_extractor(images)      # [N, D]
            
            # 2. 生成 instance-dependent 翻转概率
            w = torch.randn(features.size(1), device=device)
            logits = features @ w                      # [N]
            probs = torch.sigmoid(logits)              # (0,1)
            
            # 3. 缩放到目标噪声率 η
            if probs.mean() > 0:
                probs = probs / probs.mean() * self.percent
            probs = torch.clamp(probs, max=1.0)
            
            # 4. 按样本概率翻转标签
            noisy_labels = labels.clone()
            for i in range(len(labels)):
                if torch.rand(1, device=device).item() < probs[i]:
                    old = labels[i].item()
                    candidates = list(range(self.min_target, self.max_target + 1))
                    candidates.remove(old)
                    if candidates:
                        noisy_labels[i] = random.choice(candidates)
        
        # 计算实际噪声率
        noisy_labels_np = noisy_labels.cpu().numpy()
        actual_noise_rate = (noisy_labels_np != np.array(self.labels)).mean()
        print('noise rate = ', actual_noise_rate)
        
        # 更新标签
        self.labels = noisy_labels_np.tolist()






class NoiseMydataset(MyDataSet, NoiseDataset):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform = None,
            target_transform = None,
            mode: str = None,
            noise_type: str = 'none',
            percent: float = 0.0,
        ) -> None:

        self.transform_train_weak = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.transform_train_strong = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.transform_test = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None
        self.images_paths, self.labels = read_datalist()
        asym_trans = {
            0 : 1,
            1 : 2,
            2 : 0,

        }
        MyDataSet.__init__(self, root=root, train=train, transform=transform)
        NoiseDataset.__init__(self, noise_type=noise_type, asym_trans=asym_trans, percent=percent)

    def __getitem__(self, index):
        image = load_per_data(self.images_paths[index])
        labels = self.labels[index]
        # image, target = self.data[index], self.targets[index]
        img_array = np.array(image)
        image = Image.fromarray(img_array)

        if self.mode=='train_single':
            img = self.transform_train_weak(image)
            return img, labels
        elif self.mode=='train':
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, labels
        elif self.mode=='test':
            img = self.transform_test(image)
            return img, labels






class NoiseCIFAR10(CIFAR10, NoiseDataset):
    def __init__(
            self, 
            root: str, 
            train: bool = True,
            transform = None, 
            target_transform = None,
            download = True,
            mode: str = None,
            noise_type: str = 'none', 
            percent: float = 0.0, 
        ) -> None:

        self.transform_train_weak = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                     
            ]) 
        self.transform_train_strong = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),                
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])     

        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None
        asym_trans = {
            9 : 1, # truck ->  automobile
            2 : 0, # bird  ->  airplane
            3 : 5, # cat   ->  dog
            5 : 3, # dog   ->  cat
            4 : 7, # deer  ->  horse
        }

        CIFAR10.__init__(self, root=root, train=train, transform=transform, download=download)
        NoiseDataset.__init__(self, noise_type=noise_type, asym_trans=asym_trans, percent=percent)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.mode=='train_single':
            img = self.transform_train_weak(image)
            return img, target  
        elif self.mode=='train': 
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)   
            return raw, img1, img2, target
        elif self.mode=='test': 
            img = self.transform_test(image) 
            return img, target


class NoiseCIFAR100(CIFAR100, NoiseDataset):
    def __init__(
            self, 
            root: str, 
            train: bool = True,
            transform = None, 
            target_transform = None,
            download = True,
            mode: str = None,
            noise_type: str = 'none', 
            percent: float = 0.0, 
        ) -> None:

        self.transform_train_weak = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),                     
            ]) 
        self.transform_train_strong = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),                
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),                     
            ]) 
        self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])

        self.root = root
        self.mode = mode
        self.transform = self.transform_test
        self.target_transform = None

        CIFAR100.__init__(self, root=root, train=train, transform=transform, download=download)
        NoiseDataset.__init__(self, noise_type=noise_type, asym_trans=None, percent=percent)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.mode=='train_single':
            img = self.transform_train_weak(image)
            return img, target  
        elif self.mode=='train': 
            raw = self.transform_train_weak(image)
            img1 = self.transform_train_strong(image)
            img2 = self.transform_train_strong(image)
            return raw, img1, img2, target
        elif self.mode=='test': 
            img = self.transform_test(image) 
            return img, target



class cifar_dataloader():  
    def __init__(self, cifar_type, root, batch_size, num_workers, noise_type, percent):
        self.cifar_type = cifar_type
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_type = noise_type
        self.percent = percent

    def run(self, mode):        
        if mode=='train_single':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type, \
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type, \
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'mydataset':
                train_dataset = NoiseMydataset(root=self.root, train=True, transform=None, noise_type=self.noise_type,\
                                               percent=self.percent, mode=mode)
            else:
                raise "incorrect cifar dataset name -> (`cifar-10`, `cifar-100`)"
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=self.num_workers)           
            return train_loader 
        elif mode=='train':
            if self.cifar_type == 'cifar-10':
                train_dataset = NoiseCIFAR10(root=self.root, train=True, transform=None, noise_type=self.noise_type, \
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'cifar-100':
                train_dataset = NoiseCIFAR100(root=self.root, train=True, transform=None, noise_type=self.noise_type, \
                                                percent=self.percent, mode=mode)
            elif self.cifar_type == 'mydataset':
                train_dataset = NoiseMydataset("data\\data_csv\\train_data.csv", train=True, transform=None, noise_type=self.noise_type, \
                                               percent=self.percent, mode=mode)
            train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
                num_workers=self.num_workers)           
            return train_loader 
        elif mode=='test':
            if self.cifar_type == 'cifar-10':
                test_dataset = NoiseCIFAR10(self.root, train=False, transform=None, noise_type='none', \
                                                percent=0.0, mode=mode)
            elif self.cifar_type == 'cifar-100':
                test_dataset = NoiseCIFAR100(self.root, train=False, transform=None, noise_type='none', \
                                                percent=0.0, mode=mode)
            elif self.cifar_type == 'mydataset':
                test_dataset = NoiseMydataset(self.root, train=False, transform=None, noise_type='none',\
                                              percent=0.0, mode=mode)
                test_dataset = NoiseMydataset(root='\Paper Reproduction\\Co-learning-test2\\data\data_csv\\test_data.csv',
                                              train=False, transform=None, noise_type='none', percent=0.0, mode=mode)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                drop_last=True,
                num_workers=self.num_workers)             
            return test_loader







# class cifar_mydataloader():
#     def __init__(self, root, batch_size, num_workers, noise_type, percent):
#         self.root = root
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.noise_type = noise_type
#         self.percent = percent
#
#     def run(self, mode):
#         if mode == 'train_single':



