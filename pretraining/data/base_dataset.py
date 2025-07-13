"""
This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), 
which can be later used in subclasses.
"""

import torch.utils.data as data
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """
        Initialize the class; save the options in the class.

        Parameters
        ----------
        opt : Option class
            Stores all the experiment flags; needs to be a subclass of BaseOptions.
        """
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new dataset-specific options, and rewrite default values for existing options.

        Parameters
        ----------
        parser : ArgumentParser
            Original option parser.
        is_train : bool
            Whether training phase or test phase.
            You can use this flag to add training-specific or test-specific options.

        Returns
        -------
        parser : ArgumentParser
            The modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns
        -------
        int
            Total number of images in the dataset.
        """
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters
        ----------
        index : int
            A random integer for data indexing.

        Returns
        -------
        dict
            A dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        pass
