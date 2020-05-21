def override_dict_defaults(default_dict: dict, override_dict: dict = None) -> dict:
    if override_dict is None:
        override_dict = {}

    return {**default_dict, **override_dict}


def listify(obj: object):
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    else:
        return [obj]
