from abc import ABCMeta, abstractmethod
from typing import Any, Sized, Generator, Iterator
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict


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


class ShardedConvert(Downloader):
    def __init__(self) -> None:
        pass

    def load(self, *args, **kwargs) -> Any:
        loaded = defaultdict(list)
        for i, s in enumerate(self._load_shards(*args, **kwargs)):
            loaded[i].extend(s)
        return tuple(loaded.values())

    @abstractmethod
    def _load_shards(self, num_shards: int, *args, **kwargs) -> Generator:
        pass

    @abstractmethod
    def _process_shard(self, *args, **kwargs) -> Sized:
        pass

    @abstractmethod
    def _combine_shard_output(self, shards: Iterator) -> Any:
        pass

    @abstractmethod
    def _write(self, download_dir: str, data: Any) -> None:
        pass

    def download(self, download_dir, *args, **kwargs) -> None:
        self._write(download_dir, self._combine_shard_output(self._download_shards(*args, **kwargs)))


def convert(image_dir: str, infile: str, outfile: str, in_format: ShardedConvert, out_format: ShardedConvert,
            num_shards=100, verbose=True) -> None:
    """
    Convert annotations from one format to the other.
    :param image_dir: The directory where the images are stored.
    :param infile: The annotation file.
    :param outfile: The out file for the annotations.
    :param in_format: The current annotation format.
    :param out_format: The format in which the annotations are being converted
    :param num_shards: The number of shard to split into whne doing the conversions
    :param verbose: If to display the progress of the conversion.
    """
    shards = in_format._load_shards(num_shards, image_dir, infile)
    if verbose:
        shards = tqdm(shards, total=num_shards)

    simplified = [out_format._process_shard(*s) for s in shards]
    out_format._write(outfile, out_format._combine_shard_output(simplified))

