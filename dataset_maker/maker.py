from abc import ABC, abstractmethod
from typing import Any


class DatasetMaker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def make(self, n_samples: int) -> Any:
        NotImplementedError()
        return


class BBoxsDM(DatasetMaker):
    def __init__(self, width=640, height=640):
        super().__init__()
        self.width = width
        self.height = height

    def make(self, n_samples: int):
        NotImplementedError()
        return
