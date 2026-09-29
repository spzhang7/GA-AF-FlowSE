import logging


def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    '''
    Get logger instance
    '''

    def get_handler(handler):
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        return handler

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    if file:
        # both stdout & file
        logger.addHandler(get_handler(logging.FileHandler(name)))
        logger.addHandler(get_handler(logging.StreamHandler()))
    else:
        logger.addHandler(get_handler(logging.StreamHandler()))
    return logger
