import albumentations as alb
import cv2
import re
from functools import partial
from .constants import IMAGENET_COLOR_STD, IMAGENET_COLOR_MEAN
    
class HorizontalFlip(alb.BasicTransform):
    r"""
    Flip the image horizontally randomly (equally likely) and replace the
    word "left" with "right" in the caption.
    .. note::
        This transform can also work on images only (without the captions).
        Its behavior will be same as albumentations
        :class:`~albumentations.augmentations.transforms.HorizontalFlip`.
    Examples:
        >>> flip = HorizontalFlip(p=0.5)
        >>> out1 = flip(image=image, caption=caption)  # keys: {"image", "caption"}
        >>> # Also works with images (without caption).
        >>> out2 = flip(image=image)  # keys: {"image"}
    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply(self, img, **params):
        return cv2.flip(img, 1)

    def apply_to_caption(self, caption, **params):
        caption = (
            caption.replace("left", "[TMP]")
            .replace("right", "left")
            .replace("[TMP]", "right")
        ) 
        return caption

class VerticalFlip(alb.BasicTransform):
    r"""
    Flip the image vertically randomly (equally likely) and replace the
    word "top" with "bottom" in the caption.
    .. note::
        This transform can also work on images only (without the captions).
        Its behavior will be same as albumentations
        :class:`~albumentations.augmentations.transforms.VerticalFlip`.
    Examples:
        >>> flip = VerticalFlip(p=0.5)
        >>> out1 = flip(image=image, caption=caption)  # keys: {"image", "caption"}
        >>> # Also works with images (without caption).
        >>> out2 = flip(image=image)  # keys: {"image"}
    """

    @property
    def targets(self):
        return {"image": self.apply, "caption": self.apply_to_caption}

    def apply(self, img, **params):
        return cv2.flip(img, 0)

    def apply_to_caption(self, caption, **params):
        caption = (re.sub(r'\b(bot|bottom)\b', "top", 
                   re.sub(r'\btop\b', "[TMP]", caption)).replace("[TMP]", "bottom"))            
        return caption


class Factory:
    r"""
    Base class for all factories. All factories must inherit this base class
    and follow these guidelines for a consistent behavior:
    * Factory objects cannot be instantiated, doing ``factory = SomeFactory()``
      is illegal. Child classes should not implement ``__init__`` methods.
    * All factories must have an attribute named ``PRODUCTS`` of type
      ``Dict[str, Callable]``, which associates each class with a unique string
      name which can be used to create it.
    * All factories must implement one classmethod, :meth:`from_config` which
      contains logic for creating an object directly by taking name and other
      arguments directly from :class:`~virtex.config.Config`. They can use
      :meth:`create` already implemented in this base class.
    * :meth:`from_config` should not use too many extra arguments than the
      config itself, unless necessary (such as model parameters for optimizer).
    """

    PRODUCTS = {}

    def __init__(self):
        raise ValueError(
            f"""Cannot instantiate {self.__class__.__name__} object, use
            `create` classmethod to create a product from this factory.
            """
        )

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        r"""Create an object by its name, args and kwargs."""
        if name not in cls.PRODUCTS:
            raise KeyError(f"{cls.__class__.__name__} cannot create {name}.")

        return cls.PRODUCTS[name](*args, **kwargs)


class ImageTransformsFactory(Factory):
    r"""
    Factory to create image transformations for common preprocessing and data
    augmentations. These are a mix of default transformations from
    `albumentations <https://albumentations.readthedocs.io/en/latest/>`_ and
    some extended ones defined in :mod:`virtex.data.transforms`.
    It uses sensible default values, however they can be provided with the name
    in dict syntax. Example: ``random_resized_crop::{'scale': (0.08, 1.0)}``
    Possible choices: ``{"center_crop", "horizontal_flip", "vertical_flip", "random_resized_crop",
    "normalize", "global_resize", "color_jitter", "smallest_resize"}``.
    """

    # fmt: off
    PRODUCTS = {
        "pad_to_minimum" : partial(alb.PadIfNeeded, 
                                   border_mode = cv2.BORDER_CONSTANT, p=1.0),
        "pad_to_divisible" : partial(alb.PadIfNeeded, pad_height_divisor = 32, 
                                    pad_width_divisor = 32, 
                                    min_width = None, min_height = None, 
                                    border_mode = cv2.BORDER_CONSTANT),
        "horizontal_flip": partial(HorizontalFlip, p=0.5),
        "vertical_flip": partial(VerticalFlip, p=0.5),
        # Color normalization: whenever selected, always applied. This accepts images
        # in [0, 255], requires mean and std in [0, 1] and normalizes to `N(0, 1)`.
        "normalize": partial(
            alb.Normalize, mean=IMAGENET_COLOR_MEAN, std=IMAGENET_COLOR_STD, p=1.0
        ),
    }
    # fmt: on

    @classmethod
    def create(cls, name, *args, **kwargs):
        r"""Create an object by its name, args and kwargs."""

        if "::" in name:
            name, __kwargs = name.split("::")
            _kwargs = eval(__kwargs)
        else:
            _kwargs = {}

        _kwargs.update(kwargs)
        return super().create(name, *args, **_kwargs)
    
def create_transforms(transform_names, 
                    img_size = 512, 
                    mean = IMAGENET_COLOR_MEAN,
                    std = IMAGENET_COLOR_STD):
    image_transform_list = []

    for name in transform_names:
        # Pass dimensions if cropping / resizing, else rely on the defaults
        # as per `ImageTransformsFactory`.
        if isinstance(img_size, tuple): # model.visual.image_size is a 2-tuple
            img_size = img_size[0]
        if name == 'pad_to_minimum':
            t = ImageTransformsFactory.create(name, min_width = img_size, min_height = img_size)
        elif name == 'normalize':
            t = ImageTransformsFactory.create(name, mean = mean, std = std)
        else:
            t = ImageTransformsFactory.create(name)
        image_transform_list.append(t)

    return alb.Compose(image_transform_list)