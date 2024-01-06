# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi

"""
@author: Ridhima Garg

Introduction:
    This file contains the implementation of the main functionality of the DAWN paper.
    Use sha key and the image file to select the watermark samples.
"""


import hashlib
import hmac
import itertools
import math

import bitstring
import torch


def default_key(length: int):
    import string
    import random
    random.seed(42)
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


class WatermarkFilter:
    def __init__(self, key: str, shape: (int, int, int), precision: int = 16, probability: float = 5 / 1000, hash_f: callable = hashlib.sha256) -> None:
        self.key = key.encode("utf-8")
        self.shape = shape
        self.hash_f = hash_f
        self.precision = precision
        self.probability = probability
        self.bound = math.floor((2 ** self.precision) * self.probability)
        self.strip = 25  # None for production; used for efficiency

    @staticmethod
    def __bytes_to_bits(msg: str) -> str:
        return bitstring.BitArray(hex=msg).bin

    def __hmac(self, key: bytes, msg: bytes, strip: int = None) -> str:
        """
        creating the hash using key and msg.
        """
        if strip is not None:
            key = itertools.islice(key, 0, strip)
            msg = itertools.islice(msg, 0, strip)

        return hmac.new(key, msg, self.hash_f).hexdigest()

    def is_watermark(self, image: torch.FloatTensor) -> bool:
        # if image.shape != self.shape:
        #     raise AssertionError("Image shape {} different from expected {}.".format(image.shape, self.shape))

        hashed = self.__hmac(key=self.key, msg=image.numpy().tobytes())
        # print("hashed", hashed)
        bits = self.__bytes_to_bits(hashed)
        # print("bits", int(bits[:self.precision], 2))
        # print("bound", self.bound)
        return int(bits[:self.precision], 2) <= self.bound
    
    def label_watermark(self, image: torch.FloatTensor):

        if image.shape != self.shape:
            raise AssertionError("Image shape {} different from expected {}.".format(image.shape, self.shape))

        hashed = self.__hmac(key=self.key, msg=image.numpy().tobytes())
        # print("hashed", hashed)
        bits = self.__bytes_to_bits(hashed)

        return bits[self.precision:]


