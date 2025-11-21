"""
LRU Cache implementation for BuildAutomata Memory System
Copyright 2025 Jurden Bruce
"""

from collections import OrderedDict


class LRUCache(OrderedDict):
    """Simple LRU cache with max size"""
    def __init__(self, maxsize=1000):
        self.maxsize = maxsize
        super().__init__()

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value
