from functools import wraps


def enqueued(
    original_function=None,
    *,
    kwarg_name: str = "jobs_semaphore",
):
    """
    https://stackoverflow.com/questions/3888158/making-decorators-with-optional-arguments
    """

    def _decorate(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            semaphore = kwargs.pop(kwarg_name)
            with semaphore:
                try:
                    return function(*args, **kwargs)

                except Exception as e:
                    raise e
                finally:
                    pass

        return wrapped_function

    if original_function:
        return _decorate(original_function)

    return _decorate
