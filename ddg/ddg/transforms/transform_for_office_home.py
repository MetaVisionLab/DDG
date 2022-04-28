import torchvision.transforms
from .transforms import *
from ddg.utils import TRANSFORMS_REGISTRY

__all__ = ['TransformForOfficeHome']


@TRANSFORMS_REGISTRY.register()
class TransformForOfficeHome(Transform):

    def __init__(self, args):
        super(TransformForOfficeHome, self).__init__(args)
        self._train = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.args.input_size, self.args.input_size)),
            torchvision.transforms.RandomHorizontalFlip(),
            Random2DTranslation(self.args.input_size, self.args.input_size),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.ToTensor(),
            self.normalize,
        ])
        self._test = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.args.input_size, self.args.input_size)),
            torchvision.transforms.ToTensor(),
            self.normalize,
        ])
