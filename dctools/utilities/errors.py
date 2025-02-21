#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Classes and functions for error handling."""


class ErrorHandler:
    """Error handling."""

    @staticmethod
    def log_error(error_message: str):
        """Print a formatted error message.

        Args:
            error_message (str): Error message.
        """
        print(f"[ERROR] {error_message}")
