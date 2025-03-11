#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for error handling."""

import logging
import traceback

class DCExceptionHandler:
    """A class to handle exceptions.

    this class captures logging, and properly manages errors.
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes the exception handler with an optional logger.

        Args:
            logger (logging.Logger): A logger instance.
        """
        self.logger = logger

    def handle_exception(
            self, exception: Exception, custom_message: str = "", fail_on_error: bool=True
        ) -> None:
        """Handles an exception by logging the error and returning a formatted error message.

        Args:
            exception (Exception): The exception instance to handle.
            custom_message (str): An optional custom message to include in the log.

        Returns:
            str: A formatted error message string.
        """
        error_message = f"{custom_message}\nException: { \
            str(exception) \
        }\n{traceback.format_stack(limit=5)}"
        if fail_on_error:
            self.logger.error(error_message)
            raise(exception)
        else:
            self.logger.warning(error_message)

    def dc_raise(self, message: str) -> None:
        """Raise an exception with a custom message.

        Args:
             message (str): The custom message to include in the exception.
        """
        self.logger.error(message)
        raise Exception(message)
