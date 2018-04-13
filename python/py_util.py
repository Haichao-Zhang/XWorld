"""
This file contains some common python utility functions.
"""
import random

def overrides(interface_class):
    """
    This function is used for specifying a function in the derived class
    that is overriding an inherited class method.
    """
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


def check_or_get_value(value, valid_value_set, is_continuous=False):
    """
    Check if the given value of the specified property is a valid one, or randomly
    select one from the valid value set if value is True, and return the value.
    is_continuous denotes whenther the value is continuous (True) or discrete (False).
    """
    if not is_continuous:
        if value is None:
            assert len(valid_value_set) > 0, \
                "invalid value set %s is provided" % (valid_value_set, )
            return random.choice(valid_value_set)
        else:
            assert value in valid_value_set, \
                "input value %s is not in valid value set %s" % (value, valid_value_set)
            return value
    else:
        if value is None:
            assert len(valid_value_set) == 2 and valid_value_set[0] < valid_value_set[1], \
                "invalid value range $s is provided" % (valid_value_set, )
            return random.uniform(*valid_value_set)
        else:
            assert value >= valid_value_set[0] and value <= valid_value_set[1], \
                "input value %s is not in valid value set %s" % (value, valid_value_set)
            return value

def tsum(a, b):
    """
    perform elementwise sum for two tuples
    """
    assert isinstance(a, tuple)
    assert isinstance(b, tuple)
    assert len(a) == len(b)
    return tuple(map(sum, zip(a, b)))
