#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Functions for file handling."""

import os
import subprocess
from typing import List

from loguru import logger
from pathlib import Path
import yaml


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
            logger.warning(f"Failed to delete {file_path}: {e}")

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

def remove_listof_files(
    list_files:List[str], dir: str
) -> None:
    """Remove a list of files from a given folder.

    Args:
        list_files (List[str]): list of the files to remove
        dir (str): directory where to remove files
    """
    try:
        for filename in list_files:
            filepath = os.path.join(dir, filename)
            if os.path.isfile(filepath):
                logger.info(f"removing: {filepath}")
                os.remove(filepath)
    except Exception as exc:
        logger.error(f"Failed to remove files: {repr(exc)}")

def get_list_filter_files(
        directory: str, extension: str, regex: str, prefix: bool = False
    ) -> List[str]:
    """Return a list of files that match some conditions.

    Args:
        directory (str): directory where to look for files
        extension (str): file extension
        regex (str): pattern to match on filenames
        prefix (bool, optional): whether the filename starts with the pattern.
            Defaults to False.

    Returns:
        List[str]: _description_
    """
    list_files = list_files_with_extension(directory,  extension)
    list_filter_files = []
    if prefix:
        list_filter_files = [
            ncf for ncf in list_files if ncf.startswith(regex)
        ]
    else:
        list_filter_files = [
            ncf for ncf in list_files if regex in ncf
        ]
    return list_filter_files

def read_file_tolist(filepath: str, max_lines: int=0) -> List[str]:
    with open(filepath) as file:
        lines = []
        n_line = 0
        for line in file:
            if max_lines > 0 and n_line >= max_lines:
                break
            lines.append(line.strip())
            n_line += 1
        return lines


def check_valid_files(list_files: List[str]) -> List[str]:
    list_valid_files = []
    for file_path in list_files:
        if os.path.isfile(file_path):
            list_valid_files.append(file_path)
    return list_valid_files


def load_config_file(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)