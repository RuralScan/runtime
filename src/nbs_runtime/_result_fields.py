"""Public raster names backed by the original dataclass and pickle layout."""

from functools import wraps


def accept_baseline_keyword(stored_name):
    """Accept baseline_index with the original generated dataclass initializer."""

    def decorate(cls):
        init = cls.__init__

        @wraps(init)
        def init_with_baseline(self, *args, **kwargs):
            if "baseline_index" in kwargs:
                # dataclasses.replace supplies the old field even when the caller
                # explicitly replaces baseline_index. Prefer the explicit value.
                kwargs[stored_name] = kwargs.pop("baseline_index")
            init(self, *args, **kwargs)

        cls.__init__ = init_with_baseline
        return cls

    return decorate
