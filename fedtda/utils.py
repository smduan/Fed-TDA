import pandas as pd
from copy import deepcopy
import warnings
import importlib
import numpy as np
import contextlib
import tqdm

EPSILON = np.finfo(np.float32).eps


@contextlib.contextmanager
def set_random_state(random_state, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        random_state (int or np.random.RandomState):
            The random seed or RandomState.
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_state = np.random.get_state()

    np.random.set_state(random_state.get_state())

    try:
        yield
    finally:
        current_random_state = np.random.RandomState()
        current_random_state.set_state(np.random.get_state())
        set_model_random_state(current_random_state)
        np.random.set_state(original_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable):
            The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_state is None:
            return function(self, *args, **kwargs)

        else:
            with set_random_state(self.random_state, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper


def get_instance(obj, **kwargs):
    """Create new instance of the ``obj`` argument.

    Args:
        obj (str, type, instance):
    """
    instance = None
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        instance = getattr(importlib.import_module(package), name)(**kwargs)
    elif isinstance(obj, type):
        instance = obj(**kwargs)
    else:
        if kwargs:
            instance = obj.__class__(**kwargs)
        else:
            args = getattr(obj, '__args__', ())
            kwargs = getattr(obj, '__kwargs__', {})
            instance = obj.__class__(*args, **kwargs)

    return instance


def store_args(__init__):
    """Save ``*args`` and ``**kwargs`` used in the ``__init__`` of a copula.

    Args:
        __init__(callable): ``__init__`` function to store their arguments.

    Returns:
        callable: Decorated ``__init__`` function.
    """

    def new__init__(self, *args, **kwargs):
        args_copy = deepcopy(args)
        kwargs_copy = deepcopy(kwargs)
        __init__(self, *args, **kwargs)
        self.__args__ = args_copy
        self.__kwargs__ = kwargs_copy

    return new__init__


def get_qualified_name(_object):
    """Return the Fully Qualified Name from an instance or class."""
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


def validate_random_state(random_state):
    """Validate random state argument.

    Args:
        random_state (int, numpy.random.RandomState, tuple, or None):
            Seed or RandomState for the random generator.

    Output:
        numpy.random.RandomState
    """
    if random_state is None:
        return None

    if isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError(
            f'`random_state` {random_state} expected to be an int '
            'or `np.random.RandomState` object.')


class NotFittedError(Exception):
    """NotFittedError class."""


def validate_random_state(random_state):
    """Validate random state argument.

    Args:
        random_state (int, numpy.random.RandomState, tuple, or None):
            Seed or RandomState for the random generator.

    Output:
        numpy.random.RandomState
    """
    if random_state is None:
        return None

    if isinstance(random_state, int):
        return np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise TypeError(
            f'`random_state` {random_state} expected to be an int '
            'or `np.random.RandomState` object.')


def progress_bar_wrapper(function, pb_total, pb_description):
    """Enclose given function with a progress bar.

    Args:
        function (function):
            The function to execute.
        pb_total (int):
            The total to use in the progress bar.
        pb_description (str):
            The description of the progress bar.

    Returns:
        The function return value.
    """
    with tqdm.tqdm(total=pb_total) as progress_bar:
        progress_bar.set_description(pb_description)
        return function(progress_bar)


def handle_sampling_error(is_tmp_file, output_file_path, sampling_error):
    """Handle sampling errors by printing a user-legible error and then raising.

    Args:
        is_tmp_file (bool):
            Whether or not the output file is a temp file.
        output_file_path (str):
            The output file path.
        sampling_error:
            The error to raise.

    Side Effects:
        The error will be raised.
    """
    if 'Unable to sample any rows for the given conditions' in str(sampling_error):
        raise sampling_error

    if is_tmp_file:
        print('Error: Sampling terminated. Partial results are stored in a temporary file: '
              f'{output_file_path}. This file will be overridden the next time you sample. '
              'Please rename the file if you wish to save these results.')
    elif output_file_path is not None:
        print('Error: Sampling terminated. '
              f'Partial results are stored in {output_file_path}.')

    raise sampling_error


class ConstraintsNotMetError(ValueError):
    """Exception raised when the given data is not valid for the constraints."""

    def __init__(self, message=''):
        self.message = message
        super().__init__(self.message)


def valarray(shape, value=np.nan, typecode=None):
    """Return an array of all values.
    """

    out = np.ones(shape, dtype=bool) * value
    if typecode is not None:
        out = out.astype(typecode)
    if not isinstance(out, np.ndarray):
        out = np.asarray(out)
    return out


def argsreduce(cond, *args):
    """Return the sequence of ravel(args[i]) where ravel(condition) is
    True in 1D.
    """
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs, ]
    expand_arr = (cond == cond)
    return [np.extract(cond, arr1 * expand_arr) for arr1 in newargs]


def devide_list(data, origin_list):
    transformed_list = []
    start = 0
    for i in range(len(origin_list)):
        transformed_list.append(data[start: start+len(origin_list[i])])
        start += len(origin_list[i])

    return transformed_list
