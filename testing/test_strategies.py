import unittest
from dataset_maker.patterns import Strategies


class TestStrategies(unittest.TestCase):
    def setUp(self) -> None:
        self.strategies = Strategies()

    def test_strategies_added_and_called(self):
        key, value = 'a', lambda: 'foxyfox'
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get(key), 'foxyfox')

    def test_strategies_added_and_inputs_args(self):
        key, value = 'a', lambda x, y: x + y
        args = ('foxy', 'fox')
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get(key, *args), 'foxyfox')

    def test_strategies_added_and_inputs_kwargs(self):
        key, value = 'a', lambda x, y: x + y
        kwargs = {'y': 'foxy', 'x': 'fox'}
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get(key, **kwargs), 'foxfoxy')

    def test_strategies_added_and_inputs_args_and_kwargs(self):
        key, value = 'a', lambda x, y: x + y
        args = ('fox', )
        kwargs = {'y': 'foxy'}
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get(key, *args, **kwargs), 'foxfoxy')

    def test_strategies_add_lower_get_upper(self):
        key, value = 'a', lambda: 'foxyfox'
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get(key.upper()), 'foxyfox')

    def test_strategies_add_mix_get_different_mix(self):
        key, value = 'aBc', lambda: 'foxyfox'
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.get('ABc'), 'foxyfox')

    def test_strategies_names_stored(self):
        key, value = 'aBc', lambda: 'foxyfox'
        self.strategies.add(key, value)
        self.assertEquals(self.strategies.strategies[key.lower()][0], key)

    def test_strategies_new_same_add_updates(self):
        key, value = 'a', lambda: 'foxyfox'
        self.strategies.add(key, value)
        new_key, new_value = 'a', lambda: 'foxybox'
        self.strategies.add(new_key, new_value)
        self.assertEquals(self.strategies.get(key), 'foxybox')

    def test_strategies_new_same_add_updates_mixed_keys(self):
        key, value = 'aBC', lambda: 'foxyfox'
        self.strategies.add(key, value)
        new_key, new_value = 'abc', lambda: 'foxybox'
        self.strategies.add(new_key, new_value)
        self.assertEquals(self.strategies.get(key), 'foxybox')
