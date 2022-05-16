from numpy import tanh

def first_index(search_list, predicate):
    r"""Returns the index of the first element in the given list for which predicate is true.
    Returns None if there is no such element."""

    for index, element in enumerate(search_list):
        if predicate(element):
            return index
    return None

def contains(search_list, predicate):
    """Returns true if and only if the list contains an element x where predicate(x) is True."""

    for element in search_list:
        if predicate(element):
            return True
    return False

def coth(x): # pylint: disable=C0103
    """Hyperbolic cotangent. coth(x) = 1 / tanh(x) = cosh(x) / sinh(x)."""

    return 1 / tanh(x)
