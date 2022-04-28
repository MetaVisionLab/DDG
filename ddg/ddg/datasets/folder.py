import os
import bisect
import shutil
from pathlib import Path
from typing import List, Dict, Set
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder


class DomainFolder(ConcatDataset):
    """A generic multi-domain data loader where the images are arranged in this way by default: ::

        root/domain1/dog/xxx.png
        root/domain1/dog/xxy.png
        root/domain1/dog/[...]/xxz.png

        root/domain1/cat/123.png
        root/domain1/cat/nsdf3.png
        root/domain1/cat/[...]/asd932_.png

        root/domain2/dog/xxx.png
        root/domain2/dog/xxy.png
        root/domain2/dog/[...]/xxz.png

        root/domain2/cat/123.png
        root/domain2/cat/nsdf3.png
        root/domain2/cat/[...]/asd932_.png

    Args:
        root (string): Root directory path.
        domains (set): Domains to be loaded this time.
        splits (set): Splits to be loaded this time, e.g. 'train', 'val', 'test'.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    root: str
    domains: List
    all_domains: Dict
    domain_to_idx: Dict
    splits: List
    all_splits: Dict
    classes: List
    class_to_idx: Dict
    dataset_size: Dict

    def __init__(self, root: str, domains: Set, splits: Set, transform=None, target_transform=None, download=False):

        self.root = os.path.abspath(os.path.expanduser(root))
        self.domains = sorted(domains)
        # keep train val test order
        self.splits = [split for split in ['train', 'val', 'test'] if split in splits]

        self.check_valid_args()

        if download:
            print('process data...')
            self.download_data()
            print('Down!')

        dataset = []
        for domain in self.domains:
            for split in self.splits:
                data = ImageFolder(root=os.path.join(self.root, self.all_domains[domain], self.all_splits[split]),
                                   transform=transform,
                                   target_transform=target_transform)
                if len(data) != self.dataset_size[domain][split]:
                    raise ValueError(f'The size of {self.__class__.__name__}.{domain}.{split} is incorrect, '
                                     f'maybe some images lost, please set download to "True" to '
                                     f'reprocess the dataset.')
                dataset.append(data)

        self.classes = dataset[0].classes
        self.class_to_idx = dataset[0].class_to_idx
        self.domain_to_idx = {domain_name: i for i, domain_name in enumerate(self.domains)}

        super(DomainFolder, self).__init__(dataset)

        self.check_valid_dataset()

    def download_data(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return super(DomainFolder, self).__getitem__(idx) + self._get_domain(idx)

    def _get_domain(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return (self.domain_to_idx[self.domains[bisect.bisect_right(self.cumulative_sizes, idx) // len(self.splits)]],)

    def check_valid_args(self):
        for split in self.splits:
            assert split in self.all_splits
        for domain in self.domains:
            assert domain in self.all_domains

    def check_valid_dataset(self):
        for data in self.datasets:
            assert self.classes == data.classes
            assert self.class_to_idx == data.class_to_idx

    def _parse_split(self, split, split_file, raw_folder, domain_folder):
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                image_path = line.split(' ')[0]
                raw_path = Path(raw_folder, image_path)
                dist_path = Path(domain_folder, self.all_splits[split], image_path.split('/')[1])
                image_name = image_path.split('/')[2]
                Path(dist_path).mkdir(exist_ok=True, parents=True)
                shutil.move(raw_path, Path(dist_path, image_name))

    @property
    def num_classes(self):
        return len(self.classes)
