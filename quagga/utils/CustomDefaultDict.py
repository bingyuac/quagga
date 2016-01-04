from collections import defaultdict


class CustomDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            self[key] = self.default_factory(key)
            return self[key]
        else:
            defaultdict.__missing__(self, key)