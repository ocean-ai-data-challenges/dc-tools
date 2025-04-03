#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Handler for log messages."""

import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from json_handler import JsonHandler
from pythonjsonlogger.json import JsonFormatter
from rich.logging import RichHandler

class DCLogger(logging.Logger):
    """Custom logger class.

    A logging class to handle logs and warnings with color-coded messages.
    """

    def __init__(
            self, name: str, logfile: Optional[str] = None,
            log_level: int = logging.DEBUG,
            jsonfile: Optional[str] = None,
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
            file_handler = TimedRotatingFileHandler(
                filename=logfile, when='midnight', backupCount=7
            )
            file_format = '%(levelname)s %(asctime)s [%(filename)s:' \
                '%(funcName)s:%(lineno)d] %(message)s'
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        # JSON handler
        if jsonfile is not None:
            self.json_logger = logging.getLogger()
            self.json_logger.setLevel(logging.WARNING)
            self.jsonhandler = logging.FileHandler(jsonfile)
            json_formatter = JsonFormatter()
            self.jsonhandler.setFormatter(json_formatter)
            self.json_logger.addHandler(self.jsonhandler)


    def get_logger(self) -> logging.Logger:
        """
        Return the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger

    def get_json_logger(self) -> logging.Logger:
        """
        Return the json handler instance.

        Returns:
            JsonHandler: The json handler instance.
        """
        return self.json_logger
