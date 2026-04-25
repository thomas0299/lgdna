from functools import wraps


def logged(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import logging

        local_logger = kwargs.get("logger", logging.getLogger(func.__module__))

        if isinstance(local_logger, str):
            local_logger = logging.getLogger(local_logger)

        local_log_level = kwargs.get(
            "log_level",
            logging.INFO,
        )
        if isinstance(local_log_level, str):
            local_log_level = logging.getLevelName(local_log_level)

        local_logger.setLevel(local_log_level)

        if local_logger is None:
            kwargs["logger"] = local_logger

        local_logger.debug(
            f"Calling {func.__name__} with args: {args} and kwargs: {kwargs}"
        )

        try:
            return func(*args, **kwargs)

        except Exception as e:
            local_logger.exception(e, exc_info=True, stack_info=True)
            raise e

    return wrapper
