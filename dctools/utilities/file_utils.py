#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Functions for file handling."""

import os
import subprocess
from typing import List

from pathlib import Path


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

def list_files_with_extension(directory: str, extension: str):
    """Return a list of all files with a given extension in a specified directory.

    Args:
        directory(str): path to directory
        extension(str): file extension to look for
    """
    return [fname for fname in sorted(
        os.listdir(directory)
    ) if Path(fname).suffix == extension]

def delete_files_from_list(directory: str, list_files: List[str]):
    """Remove a list of files in a given directory.

    directory(str): directory
    list_files(List[str]): list of files to delete
    """
    for fname in list_files:
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

def run_command(command: str):
    """Run and wait till the end of the given command.

    Args:
        command(str): command to run
    """
    cmd = [command]
    p = subprocess.Popen(cmd)
    p.wait()
    return p.returncode
