import os
import shutil
from pathlib import Path
from .folder import DomainFolder
from ddg.utils import DATASET_REGISTRY
from torchvision.datasets.utils import extract_archive, download_url


@DATASET_REGISTRY.register()
class DomainNet(DomainFolder):
    """The DomainNet multi-domain data loader

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/, cleaned version.

    Special note: 1. The 6th and 7th data of syringe are missing in painting_train.txt;
                  2. The syringe class (301) is missing in painting_test.txt;
                  3. The 6th and 7th data of t-shirt are missing in painting_test.txt;
                  4. The t-shirt class (328) is missing in painting_train.txt;
                  5. We have handled this situation, please refer to the fix_error function for more details.

    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    all_domains = {'Clipart': 'clipart',
                   'Infograph': 'infograph',
                   'Painting': 'painting',
                   'Quickdraw': 'quickdraw',
                   'Real': 'real',
                   'Sketch': 'sketch'
                   }
    all_splits = {'train': 'train',
                  'test': 'test'
                  }
    dataset_size = {'Clipart': {'train': 33525, 'test': 14604},
                    'Infograph': {'train': 36023, 'test': 15582},
                    'Painting': {'train': 50426, 'test': 21855},
                    'Quickdraw': {'train': 120750, 'test': 51750},
                    'Real': {'train': 120906, 'test': 52041},
                    'Sketch': {'train': 48212, 'test': 20916}
                    }

    def __init__(self, root, domains, splits, transform=None, target_transform=None, download=False):

        root = os.path.join(root, 'domainnet')

        if 'val' in splits:
            splits.remove('val')
            splits.add('test')

        super(DomainNet, self).__init__(root=root,
                                        domains=domains,
                                        splits=splits,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)

    def download_data(self):

        Path(self.root, 'splits').mkdir(exist_ok=True, parents=True)

        resources = [
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
             "clipart.zip", "cd0d8f2d77a4e181449b78ed62bccf1e"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
             "infograph.zip", "720380b86f9e6ab4805bb38b6bd135f8"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
             "painting.zip", "1ae32cdb4f98fe7ab5eb0a351768abfd"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
             "quickdraw.zip", "bdc1b6f09f277da1a263389efe0c7a66"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
             "real.zip", "dcc47055e8935767784b7162e7c7cca6"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
             "sketch.zip", "658d8009644040ff7ce30bb2e820850f"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
             "splits/clipart_train.txt", "b4349693a7f9c05c53955725c47ed6cb"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
             "splits/infograph_train.txt", "379b50054f4ac2018dca4f89421b92d9"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
             "splits/painting_train.txt", "7db0e7ca73ad9982f6e1f7f3ef372c0a"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
             "splits/quickdraw_train.txt", "b732ced3939ac8efdd8c0a889dca56cc"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
             "splits/real_train.txt", "8ebf02c2075fadd564705f0dc7cd6291"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
             "splits/sketch_train.txt", "1233bd18aa9a8a200bf4cecf1c34ef3e"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt",
             "splits/clipart_test.txt", "f5ddbcfd657a3acf9d0f7da10db22565"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt",
             "splits/infograph_test.txt", "779626b50869edffe8ea6941c3755c71"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt",
             "splits/painting_test.txt", "232b35dc53f26d414686ae579e38d9b5"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt",
             "splits/quickdraw_test.txt", "c1a828fdfe216fb109f1c0083a252c6f"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt",
             "splits/real_test.txt", "6098816791c3ebed543c71ffa11b9054"),
            ("http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt",
             "splits/sketch_test.txt", "d8a222e4672cfd585298aa14d02ea441")
        ]
        for url, filename, md5 in resources:
            download_url(url, root=self.root, filename=filename, md5=md5)

        for domain in self.all_domains:
            domain_folder = Path(self.root, self.all_domains[domain])
            raw_folder = Path(str(domain_folder) + '_raw')
            if domain_folder.exists():
                shutil.rmtree(domain_folder)
            if raw_folder.exists():
                shutil.rmtree(raw_folder)
            extract_archive(str(domain_folder) + '.zip', to_path=str(raw_folder))
            for split in ['train', 'test']:
                split_file = Path(self.root, 'splits', self.all_domains[domain] + '_' + self.all_splits[split] + '.txt')
                self._parse_split(split=split, split_file=split_file, raw_folder=raw_folder,
                                  domain_folder=domain_folder)
            if domain == 'Painting':
                self.fix_error(raw_folder=raw_folder, domain_folder=domain_folder)
            shutil.rmtree(raw_folder)

    @staticmethod
    def fix_error(raw_folder, domain_folder):
        error_file = [
            (os.path.join(raw_folder, 'painting', 'syringe', 'painting_301_000006.jpg'),
             os.path.join(domain_folder, 'train', 'syringe', 'painting_301_000006.jpg')),
            (os.path.join(raw_folder, 'painting', 'syringe', 'painting_301_000007.jpg'),
             os.path.join(domain_folder, 'train', 'syringe', 'painting_301_000007.jpg')),
            (os.path.join(raw_folder, 'painting', 'syringe'),
             os.path.join(domain_folder, 'test', 'syringe')),
            (os.path.join(raw_folder, 'painting', 't-shirt', 'painting_328_000006.jpg'),
             os.path.join(domain_folder, 'test', 't-shirt', 'painting_328_000006.jpg')),
            (os.path.join(raw_folder, 'painting', 't-shirt', 'painting_328_000007.jpg'),
             os.path.join(domain_folder, 'test', 't-shirt', 'painting_328_000007.jpg')),
            (os.path.join(raw_folder, 'painting', 't-shirt'),
             os.path.join(domain_folder, 'train', 't-shirt'))
        ]
        for src, dst in error_file:
            shutil.move(src, dst)
