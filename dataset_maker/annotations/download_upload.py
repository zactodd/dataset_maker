from abc import ABCMeta, abstractmethod


class Loader(metaclass=ABCMeta):
    """
    Command pattern class for loading annotation data.
    """
    def __init__(self):
        pass

    @abstractmethod
    def load(self):
        pass


class Downloader(metaclass=ABCMeta):
    """
    Command pattern class for loading annotation data.
    """
    def __init__(self):
        pass

    @abstractmethod
    def download(self):
        pass


class LoaderDownloader(Loader, Downloader, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

