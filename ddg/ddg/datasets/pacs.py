import os
import shutil
from pathlib import Path
from .folder import DomainFolder
from ddg.utils import DATASET_REGISTRY
from torchvision.datasets.utils import download_and_extract_archive


@DATASET_REGISTRY.register()
class PACS(DomainFolder):
    """The VLCS multi-domain data loader

    Statistics:
        - 4 domains: ArtPainting, Cartoon, Photo, Sketch
        - 5 categories: bird, car, chair, dog, and person.
        - URL: https://dali-dl.github.io/project_iccv2017.html.

    Reference:
        - Da Li et al. Deeper, Broader and Artier Domain Generalization. ICCV 2017.
    """

    all_domains = {'ArtPainting': 'art_painting',
                   'Cartoon': 'cartoon',
                   'Photo': 'photo',
                   'Sketch': 'sketch'
                   }
    all_splits = {'train': 'train',
                  'val': 'crossval'
                  }
    dataset_size = {'ArtPainting': {'train': 1840, 'val': 208},
                    'Cartoon': {'train': 2107, 'val': 237},
                    'Photo': {'train': 1499, 'val': 171},
                    'Sketch': {'train': 3531, 'val': 398}
                    }

    def __init__(self, root, domains, splits, transform=None, target_transform=None, download=False):

        root = os.path.join(root, 'PACS')

        if 'test' in splits:
            splits.remove('test')
            splits.add('train')
            splits.add('val')

        super(PACS, self).__init__(root=root,
                                   domains=domains,
                                   splits=splits,
                                   transform=transform,
                                   target_transform=target_transform,
                                   download=download)

    def download_data(self):

        raw_folder = Path(self.root, 'kfold')
        split_folder = Path(self.root, 'splits')
        if raw_folder.exists():
            shutil.rmtree(raw_folder)
        if split_folder.exists():
            shutil.rmtree(split_folder)

        resources = [
            ("https://1h6pna.bn.files.1drv.com/y4mB9wf3RbCy6fT22q2y-1B--AA3TupRF_i47kEM9jLwUOfvKAq"
             "4XV4F7qzJLVMocveHN_57JkOCXkoFfY0QxLBZ7sHkNKqe5xn-XmKSnpmwlLHEggeoGQth16r1TeMPk4NCZ2w"
             "97KTIV1u1J0k496SfioRtg6LU0CIE8FZT67W51nqejEU9_qjisfTROALVihzrnWuV1NihwQV4yZK8tfc0w",
             "PACS.zip", "07272ddb516cd98cc7b044af74b51ab9")
        ]

        for url, filename, md5 in resources:
            download_and_extract_archive(url, download_root=self.root, filename=filename, md5=md5)

        for domain in self.all_domains:
            domain_folder = Path(self.root, self.all_domains[domain])
            if domain_folder.exists():
                shutil.rmtree(domain_folder)
            for split in ['train', 'val']:
                split_file = Path(split_folder, self.all_domains[domain] + '_' + self.all_splits[split] + '_kfold.txt')
                self._parse_split(split=split, split_file=split_file,
                                  raw_folder=raw_folder, domain_folder=domain_folder)
        shutil.rmtree(raw_folder)
