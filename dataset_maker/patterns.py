from abc import ABCMeta
from types_utils import F
from typing import Type


class Singleton(type):
    """
    Define an instance operation that lets clients access its unique
    instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class PolymorphicSingleton(ABCMeta, Singleton):
    """
    Defines a polymorphic instance operation that lets clients access its unique
    instance.
    """
    def __init__(cls, name, bases, attrs, **kwargs):
        super().__init__(name, bases, attrs, **kwargs)


class Strategies:
    def __init__(self):
        self.strategies = {}

    def add(self, k: str, class_reference: object) -> None:
        """
        Use this class to add a method to the image loader.
        :param k: The key to set the ImageLoader class.
        :param class_reference: the class of the image loader.
        """
        self.strategies[k] = class_reference

    def get(self, k: str, **kwargs) -> object:
        """
        Get the strategy class and initialised an instance of that class.
        :param k: The key for the strategy class.
        :param kwargs: Extra class specific information.
        :return: The initialised strategy class.
        """
        return self.strategies[k](**kwargs)


class SingletonStrategies(Strategies, metaclass=Singleton):
    def __init__(self):
        super().__init__()


def strategy_method(parent: Type[SingletonStrategies], name: str = None) -> F:
    assert isinstance(parent, SingletonStrategies) or issubclass(parent, SingletonStrategies)

    def inner(cls):
        parent().add(cls.__name__ if name is None else name, cls)
        return cls
    return inner
