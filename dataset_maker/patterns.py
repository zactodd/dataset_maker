from abc import ABCMeta
from dataset_maker.types_utils import F
from typing import Type, Any


class Singleton(type):
    """
    Define an instance operation that lets clients access its unique
    instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Strategies:
    def __init__(self):
        self.strategies = {}

    def add(self, name: str, class_reference: object) -> None:
        """
        Use this class to add a strategy to Strategies class .
        :param name: The name of the strategy the lowercase value of the will act as the key.
        :param class_reference: the class of the image loader.
        """
        self.strategies[name.lower()] = (name, class_reference)

    def get(self, name: str, *args, **kwargs) -> Any:
        """
        Get the strategy class and initialised an instance of that class.
        :param name: The name of the strategy the lowercase value of the name is the key.
        :param kwargs: Extra class specific information.
        :return: The initialised strategy class.
        """
        return self.strategies[name.lower()][1](*args, **kwargs)


class SingletonStrategies(Strategies, metaclass=Singleton):
    def __init__(self):
        super().__init__()


def strategy_method(parent: Type[SingletonStrategies], name: str = None) -> F:
    assert isinstance(parent, SingletonStrategies) or issubclass(parent, SingletonStrategies)

    def inner(cls):
        parent().add(cls.__name__ if name is None else name, cls)
        return cls
    return inner


class Register(metaclass=ABCMeta):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    def __new__(cls, *args, **kwargs):
        subclass = cls._registry[cls.__name__]
        obj = object.__new__(subclass)
        return obj
