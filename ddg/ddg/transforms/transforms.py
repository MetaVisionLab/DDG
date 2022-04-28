import random
from PIL import Image
import torchvision

__all__ = ['Transform', 'Random2DTranslation']


class Transform:
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    _train: torchvision.transforms.Compose
    _test: torchvision.transforms.Compose

    def __init__(self, args):
        super(Transform, self).__init__()
        self.args = args

    @property
    def train(self):
        return self._train

    @property
    def test(self):
        return self._test


class Random2DTranslation:
    """Randomly translates the input image with a probability.
    Specifically, given a predefined shape (height, width), the
    input is first resized with a factor of 1.125, leading to
    (height*1.125, width*1.125), then a random crop is performed.
    Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )

        return croped_img
