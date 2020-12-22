from abc import ABCMeta, abstractmethod
from typing import Any


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
    def download(self, *args, **kwargs) -> None:
        pass


class LoaderDownloader(Loader, Downloader, metaclass=ABCMeta):
    """
    Abstract base class for loading and downloading annotation data.
    """
    def __init__(self) -> None:
        super().__init__()

