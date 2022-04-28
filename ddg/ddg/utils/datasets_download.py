import unittest
from ddg.datasets import *


class TestDatasetsDownload(unittest.TestCase):
    ROOT = './data'

    def test_PACS_download(self):
        print('test PACS dataset download')
        PACS(root=self.ROOT,
             domains={'ArtPainting', 'Cartoon', 'Photo', 'Sketch'},
             splits={'train', 'val', 'test'},
             download=True)

    def test_OfficeHome(self):
        print('test OfficeHome dataset download')
        OfficeHome(root=self.ROOT,
                   domains={'Art', 'Clipart', 'Product', 'RealWorld'},
                   splits={'train', 'val', 'test'},
                   download=True)

    def test_DomainNet_download(self):
        print('test DomainNet dataset')
        DomainNet(root=self.ROOT,
                  domains={'Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch'},
                  splits={'train', 'val', 'test'},
                  download=True)


if __name__ == '__main__':
    unittest.main()
