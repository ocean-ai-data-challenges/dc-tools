#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Handler for log messages."""

import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from rich.logging import RichHandler

class DCLogger(logging.Logger):
    """Custom logger class.

    A logging class to handle logs and warnings with color-coded messages.
    """

    def __init__(
            self, name: str, logfile: Optional[str] = None, log_level: int = logging.DEBUG
        ) -> None:
        """
        Initialize a logger with console and file handlers.

        Args:
            name (str): The name of the logger.
            log_level (int): The logging level (default: DEBUG).
        """
        super().__init__(name, log_level)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Console handler
        console_handler = RichHandler()
        fmt_console = '%(asctime)s - %(message)s'
        console_formatter = logging.Formatter(fmt_console)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if logfile is not None:
            # file_handler = logging.FileHandler(logfile, 'w+')
            file_handler = TimedRotatingFileHandler(
                filename=logfile, when='midnight', backupCount=7
            )

            file_format = '%(levelname)s %(asctime)s [%(filename)s:' \
                '%(funcName)s:%(lineno)d] %(message)s'
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Return the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    '''@staticmethod
    def info(message: str, stacklevel=2, *args, **kwargs) -> None:
        """
        Log an informational message with blue color.

        Args:
            message (str): The message to log.
        """
        logging.info(message)

    #@staticmethod
    def warning(self, message: str, stacklevel=2, *args, **kwargs) -> None:
        """
        Log a warning message with yellow color.

        Args:
            message (str): The message to log.
        """
        logging.warning(message)

    #@staticmethod
    def error(self, message: str, stacklevel=2, *args, **kwargs) -> None:
        """
        Log an error message with red color.

        Args:
            message (str): The message to log.
        """
        logging.error(message)

    #@staticmethod
    def debug(self, message: str, stacklevel=2, *args, **kwargs) -> None:
        """
        Log a debug message with green color.

        Args:
            message (str): The message to log.
        """
        logging.debug(message)'''
