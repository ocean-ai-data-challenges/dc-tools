#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Functions for file handling."""

import os


def empty_folder(folder_path: str):
    """Remove all files in given folder.

    Args:
            folder_path (str): Path to ,folder to empty.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
