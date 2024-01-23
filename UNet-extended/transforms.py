#!/usr/bin/env python
# coding: utf-8


import numpy as np


class ToUint8(object):
    def __call__(self, pic):
        """
        Returns:
            array: Converted image.
        """
        return np.array(pic).astype(np.uint8)

    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToFloat32(object):
    def __call__(self, pic):
        """
        Returns:
            array: Converted image.
        """
        return np.array(pic).astype(np.float32)

    
    def __repr__(self):
        return self.__class__.__name__ + '()'


    
class DivideBy(object):
    """
    Sometimes we need for instance to divide by 255 in order to scale the input btw. 0 and 1.
    """

    def __init__(self, divisor):
        self.divisor = divisor


    def __call__(self, pic):
        """
        Returns:
            array: Converted image.
        """
        return pic/self.divisor
    
    
    def __repr__(self):
        return self.__class__.__name__ + '()'

    
class DivideBy255(DivideBy):
    """
    Sometimes we need to divide by 255 in order to scale the input btw. 0 and 1.
    """

    def __init__(self):
        super().__init__(255)

    
    def __repr__(self):
        return self.__class__.__name__ + '()'

    
class ToNumpy(object):
    """Convert a ``Tensor`` to numpy array.
    This is just for convenience in case one would prefer to work with numpy arrays for some reason.
    """

    def __call__(self, pic):
        """
        Args:
            pic (Tensor): Image to be converted to numpy array.
        Returns:
            array: Converted image.
        """
        return pic.numpy()

    
    def __repr__(self):
        return self.__class__.__name__ + '()'

    
# todo: provide transform instead of handling this inside the dataset implementation
class FlipLRAugment(object):
    def __call__(self, pic):
        """
        Returns:
            array: Converted image.
        """
        if random.random() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        return np.array(pic).astype(np.uint8)
    
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
    