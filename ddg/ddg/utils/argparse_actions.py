import argparse
from collections import OrderedDict


__all__ = ['DictAction']


class DictAction(argparse.Action):
    """DictAction: argparse can get KEY=VALUE pair inputs, args will get an OrderedDict.

    Note: This class will try best effort to parse values to int or float.

    """
    def __call__(self, parser, namespace, values, option_string=None):
        kv = OrderedDict()
        for value in values:
            assert value.count('=') == 1, f'value of {namespace} must be KEY=VALUE pairs, but got "{value}".'
            k, v = value.split('=')
            v = self.int_or_float(v) if self.is_int_or_float(v) else v
            kv[k] = v
        setattr(namespace, self.dest, kv)

    @staticmethod
    def is_int_or_float(value):
        try:
            int(value)
        except ValueError:
            try:
                float(value)
            except ValueError:
                return False
        return True

    @staticmethod
    def int_or_float(value):
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f'{value} can not convert to int or float.')
        return value
