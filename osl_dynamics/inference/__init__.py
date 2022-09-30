"""Functions for inference.

"""

import importlib


def lazy_import(importer_name, to_import):
    """Return the importing module and a callable for lazy importing.

    The module named by importer_name represents the module performing the
    import to help facilitate resolving relative imports.

    to_import is an iterable of the modules to be potentially imported (absolute
    or relative). The `as` form of importing is also supported,
    e.g. `pkg.mod as spam`.

    This function returns a tuple of two items. The first is the importer
    module for easy reference within itself. The second item is a callable to be
    set to `__getattr__`.
    """
    module = importlib.import_module(importer_name)
    import_mapping = {}
    for name in to_import:
        importing, _, binding = name.partition(" as ")
        if not binding:
            _, _, binding = importing.rpartition(".")
        import_mapping[binding] = importing

    def __getattr__(name):
        if name not in import_mapping:
            message = f"module {importer_name!r} has no attribute {name!r}"
            raise AttributeError(message)
        importing = import_mapping[name]
        # imortlib.import_module() implicitly sets submodules on this module as
        # appropriate for direct imports.
        imported = importlib.import_module(importing, module.__spec__.parent)
        setattr(module, name, imported)
        return imported

    return module, __getattr__


mod, __getattr__ = lazy_import(
    __name__,
    {
        "osl_dynamics.inference.callbacks",
        "osl_dynamics.inference.initializers",
        "osl_dynamics.inference.layers",
        "osl_dynamics.inference.metrics",
        "osl_dynamics.inference.modes",
        "osl_dynamics.inference.regularizers",
        "osl_dynamics.inference.tf_ops",
    },
)
