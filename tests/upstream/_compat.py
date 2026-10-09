"""Minimal stand-in for the parts of ``nose.tools`` the upstream suite used
(nose does not run on Python >= 3.10)."""
import functools

import pytest


def raises(*exceptions):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with pytest.raises(exceptions):
                fn(*args, **kwargs)
        return wrapper
    return decorate
