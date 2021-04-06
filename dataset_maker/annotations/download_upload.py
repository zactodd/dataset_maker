from abc import ABCMeta, abstractmethod
from typing import Any
import os
import numpy as np


class Loader(metaclass=ABCMeta):
    """
    Command pattern class for loading annotation data.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        pass


class Downloader(metaclass=ABCMeta):
    """
    Command pattern class for download annotation data.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def download(self, download_dir, *args, **kwargs) -> None:
        pass

    def split_download(self, download_dir, *args, ratios=(0.8, 0.1, 0.1), spilt_names=("train", "val", "test"),
                       **kwargs):
        assert len(ratios) == len(spilt_names), \
            "There needs to be the same number of names as split ratios."
        assert all(len(args[0]) == len(a) for a in args[1:]), "all args must have the same length."

        # Creating directories for split files
        paths = []
        for n in spilt_names:
            path = f"{download_dir}/{n}"
            if not os.path.exists(path):
                os.mkdir(path)
            paths.append(path)

        # Creating split indices
        num_args = len(args[0])
        values = [int((sum(ratios[:i]) + a) * num_args) for i, a in enumerate(ratios)]
        idx = np.random.permutation(num_args)
        splits = np.split(idx, values)

        # Saving split files
        indexable_args = tuple(zip(*args))
        for path, s in zip(paths, splits):
            self.download(path, *zip(*(indexable_args[i] for i in s)), **kwargs)


class LoaderDownloader(Loader, Downloader, metaclass=ABCMeta):
    """
    Abstract base class for loading and downloading annotation data.
    """
    def __init__(self) -> None:
        super().__init__()


