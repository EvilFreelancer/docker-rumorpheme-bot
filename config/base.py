import os


class ImproperlyConfigured(Exception):
    """Raises when a environment variable is missing."""
    def __init__(self, variable_name: str, *args, **kwargs):
        self.variable_name = variable_name
        self.message = f"Set the {variable_name} environment variable."
        super().__init__(self.message, *args, **kwargs)


def getenv(var_name: str, cast_to=str, default=None):
    """Gets an environment variable or raises an exception."""
    try:
        value = os.getenv(var_name, default)
        if cast_to == list:
            return value.split(',')
        return cast_to(value)
    except KeyError:
        raise ImproperlyConfigured(var_name)
    except ValueError:
        raise ValueError(f"The value {value} can't be cast to {cast_to}.")
