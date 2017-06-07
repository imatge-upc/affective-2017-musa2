"""
Small library that points to the MT-VSO data set.
"""
from __future__ import absolute_import

from util.dataset import Dataset


class MTVSOData(Dataset):
    """
    MVSO dataset
    """

    def __init__(self, subset):
        super(MTVSOData, self).__init__('MTVSO', subset)

    def num_classes(self):
        """Returns the number of classes in the data set as [ANPs, Noun, Adjectives]."""
        return 553, 167, 117

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return 307185
        if self.subset == 'val':
            return 77073

    def download_message(self):
        pass
