import functools
import logging

import time
from typing import Optional

logger = logging.getLogger(__name__)

"""
Some handy decorators or sudo decorators for time series models
"""


def domain_wrangler(
    func: Optional[callable] = None, stash_domain: Optional[str] = None
):
    """
    Decorator for RegularTimeSeriesModel instance methods that turns
    start end and locations into a numpy model domain as the argument to the wrapped method.
    Can be used as a decorator with or without a calling domain_wrangler as a function due to partial magic

    @domain_wrangler
    def myfunc(self, domain, *args, **kwargs):
        pass

    or

    @domain_wrangler(stash_domain="_fit_domain_args")
    def myfunc(self, domain, *args, **kwargs):
        pass

    The stash_domain argument is used to persist the start, end and locations arguments on the instance.
    This is much better than storing the much larger actual numpy domain.

    :param func: the callable instance method to wrap
    :param stash_domain: the instance variable to stash the start, stop and locations in as a tuple
    :return: the wrapped method
    """
    if func is None:
        # if called as a function `@domain_wrangler(stash_domain="foo")` use partial to return a new function
        # as the decorator while passing the stash_domain argument via partial.
        return functools.partial(domain_wrangler, stash_domain=stash_domain)

    @functools.wraps(func)
    def wrapper_decorator(self, start, end, *locations, **kwargs):
        result = func(self, self.domain(start, end, *locations), **kwargs)
        if stash_domain:
            setattr(self, stash_domain, (start, end, locations))
        return result

    return wrapper_decorator


def remove_custom_kwargs(func):
    @functools.wraps(func)
    def wrapper_decorator(self, **kwargs):
        return func(
            self, **{k: kwargs[k] for k in kwargs if k not in self.CUSTOM_KWARGS}
        )

    return wrapper_decorator


def feature_names(names):
    """
    Not a normal decorator - it returns the original function object with a method attached to it.
    :param names: the list of feature names to return for this function in a FunctionTransformer
    :return: the parameterized decorator
    """

    def gfn():
        return names

    def decorator(func):
        func.get_feature_names = gfn
        return func

    return decorator


def method_binder(method, instance, as_name=None):
    as_name = as_name or method.__name__
    bound = method.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound)


def function_string_binder(func, name=None):
    def __str__(self):
        if name:
            return name
        else:
            return func.__name__

    method_binder(__str__, func)


def feature_names_wrapper(prefix, previous_transform):
    """
    Normal decorator - but typically called inline inside a pipeline to wrap a method in a particular context
    ## This application breaks pickle! ##
    Consider removing? The artifact is here more as a record of what I have tried and didn't work.
    :param prefix: the prefix for the feature name
    :param previous_transform: the previous transform to get feature names from
    :return: the decorator of the function
    """

    def decorator(func):
        def get_feature_names(self):
            return [
                "{}_{}".format(prefix, name)
                for name in previous_transform.get_feature_names()
            ]

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        method_binder(get_feature_names, wrapper)

        return wrapper

    return decorator


def function_string_decorator(name=None):
    """
    Normal decorator - it returns the original function object with __str__ implemented.
    :param name: optional name if the default is not desired
    :return: the decorator
    """

    def decorator(func):
        def __str__(self):
            if name:
                return name
            else:
                return func.__name__

        method_binder(__str__, func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def timer(log=None):
    """
    Logging timer data is available but should be avoided in production code.
    Use prometheus metrics instead.
    :param log:
    :return:
    """
    if log is None:
        log = logger

    def decorator(func):
        """Print the runtime of the decorated function"""

        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            # TODO handle unbound methods
            log.info(
                f"Finished {args[0].__class__.__name__!r} {func.__name__!r} in {run_time:.4f} secs"
            )
            return value

        return wrapper_timer

    return decorator


def debug(log=None):
    if log is None:
        log = logger

    def decorator(func):
        """Print the function signature and return value"""

        @functools.wraps(func)
        def wrapper_debug(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            log.debug(f"Calling {func.__name__}({signature})")
            value = func(*args, **kwargs)
            log.debug(f"{func.__name__!r} returned {value!r}")
            return value

        return wrapper_debug

    return decorator


def escape_context_property(name: str):
    """
    This decorator enables bypassing the context manager by calling __enter__ via a dynamically assigned property.
    In a cell-based execution environment such as a jupyter notebook, this allows the connection to a resource to stay open and to persist across cells.

    When using a class that implements context manager this should not be necessary. For instance, it is up to
    the developer whether to use `f=open(...)` or `with open(...) as f`. However, when using the very helpful
    context manager decorator to make a function into a context manger it is difficult to get out of the
    context manager pattern, for example in a cell based execution environment like jupyter notebook. If the
    developer wishes to hold the resource open between cells as you might with a file or connection, the developer
    can only do this by calling `__enter__()` on the object which is a kludge. This decorator does this for you in a
    explicit way. While this can be used as a decorator, it can also be called directly on the function if we
    don't want to modify the function in other use cases.

    Normal decorator use pattern:
        @escape_context_property("the_prop")
        @contextlib.contextmanager
        def foo(...):
            ...
            yield something

    Still allows:
        with foo(...) as context:
            ...
    Or, to hold the resource open...
        context = foo(...).the_prop
        ...

    You can also apply this decorator explicitly only when you want to use the property pattern:
        @contextlib.contextmanager
        def foo(...):
            ...
            yield something

        foo_hack = escape_context_property("the_prop")
        context = foo_hack(...).the_prop

    At the time of implementation, this is intended for use in Jupyter Notebooks. It is expected we may add the
    decorator to functions used in production, but the usage and pitfalls should be clearly documented. It is not
    anticipated that the context manager will actually be subverted using this pattern in production code.

    :param name: The name of the property to use
    :return: the decorator
    """

    def decorator(func):
        def attach_dyn_propr(instance, prop_name, propr):
            """Attach property proper to instance with name prop_name.
            Reference:
              * https://stackoverflow.com/a/1355444/509706
              * https://stackoverflow.com/questions/48448074
            """
            class_name = instance.__class__.__name__ + "Child"
            child_class = type(class_name, (instance.__class__,), {prop_name: propr})
            instance.__class__ = child_class

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            attach_dyn_propr(result, name, property(lambda self: self.__enter__()))
            return result

        return wrapper

    return decorator
