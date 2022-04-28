import os
import math
import shutil
import numpy as np
from pathlib import Path
from .folder import DomainFolder
from ddg.utils import DATASET_REGISTRY
from torchvision.datasets.utils import download_and_extract_archive


@DATASET_REGISTRY.register()
class OfficeHome(DomainFolder):
    """The OfficeHome multi-domain data loader

    Statistics:
        - Around 15,500 images.
        - 65 classes related to office and home objects.
        - 4 domains: Art, Clipart, Product, RealWorld.
        - URL: https://www.hemanthdv.org/officeHomeDataset.html.

    Reference:
        - Venkateswara et al. Deep Hashing Network for Unsupervised
        Domain Adaptation. CVPR 2017.
    """

    all_domains = {'Art': 'Art',
                   'Clipart': 'Clipart',
                   'Product': 'Product',
                   'RealWorld': 'RealWorld'
                   }
    all_splits = {'train': 'train',
                  'val': 'val'
                  }
    dataset_size = {'Art': {'train': 2162, 'val': 265},
                    'Clipart': {'train': 3909, 'val': 456},
                    'Product': {'train': 3969, 'val': 470},
                    'RealWorld': {'train': 3892, 'val': 465}
                    }
    split_ratio = 0.9

    def __init__(self, root, domains, splits, transform=None, target_transform=None, download=False):

        root = os.path.join(root, 'office_home')

        if 'test' in splits:
            splits.remove('test')
            splits.add('train')
            splits.add('val')

        super(OfficeHome, self).__init__(root=root,
                                         domains=domains,
                                         splits=splits,
                                         transform=transform,
                                         target_transform=target_transform,
                                         download=download)

    def download_data(self):

        raw_folder = Path(self.root, 'OfficeHomeDataset_10072016')
        if raw_folder.exists():
            shutil.rmtree(raw_folder)

        resources = [
            ("https://d6rf5q.bn.files.1drv.com/y4mkRt2hJeKo_6wiEzYlAix3uVv3YoIHzLwtG1f_pQCZKumi1b"
             "oTZZUDdiHTLre7X4Gb6e28yxMLRSb1WMXCc5uBGKhpnRg6vP3O55hH10Lirp3alnuas1kml_lP3YQ82s9sp"
             "JUj6HsIfaLedSy-VX1LD70-0_i_VlA2_fwUIsRzepx5NMZWVRr7lWiTpNps8ysfUrKppMg1ZEHW0vnvAHr7A",
             "OfficeHomeDataset_10072016.zip", "b1c14819770c4448fd5b6d931031c91c")
        ]
        for url, filename, md5 in resources:
            download_and_extract_archive(url, download_root=self.root, filename=filename, md5=md5)

        shutil.move(Path(raw_folder, 'Real World'), Path(raw_folder, 'RealWorld'))

        for domain in self.all_domains:
            domain_folder = Path(self.root, self.all_domains[domain])
            raw_domain_folder = Path(raw_folder, self.all_domains[domain])
            if domain_folder.exists():
                shutil.rmtree(domain_folder)
            domain_folder.mkdir(exist_ok=True, parents=True)
            for raw_data_folder in raw_domain_folder.iterdir():
                category = raw_data_folder.name
                train_folder = Path(domain_folder, self.all_splits['train'], category)
                val_folder = Path(domain_folder, self.all_splits['val'], category)
                train_folder.mkdir(exist_ok=True, parents=True)
                val_folder.mkdir(exist_ok=True, parents=True)
                images = os.listdir(raw_data_folder)
                images_number = len(images)
                permutation = np.random.permutation(images_number)
                split = math.floor(images_number * self.split_ratio)
                for i in range(split):
                    shutil.move(Path(raw_data_folder, images[permutation[i]]),
                                Path(train_folder, images[permutation[i]]))
                for i in range(split, images_number):
                    shutil.move(Path(raw_data_folder, images[permutation[i]]),
                                Path(val_folder, images[permutation[i]]))
        shutil.rmtree(raw_folder)
