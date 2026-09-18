"""The only name mapping. Modules are imported on selection, not on package import."""

from importlib import import_module

METHODS = {"linear": "methods.linear", "ragged": "methods.ragged"}


def get(name):
    return import_module(METHODS[name])
