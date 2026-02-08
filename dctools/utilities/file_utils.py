# -*- coding: UTF-8 -*-

"""Functions for file handling."""

from collections import OrderedDict
import os
import subprocess
from typing import Any, Dict, List, Optional

from loguru import logger
from pathlib import Path
import yaml


def remove_file(filepath: str) -> bool:
    """
    Remove a local file.

    Args:
        filepath (str): Path of the file to remove.

    Returns:
        bool: True if the file was deleted, False otherwise.
    """
    try:
        if not isinstance(filepath, str) or not filepath:
            logger.warning(f"remove_file: invalid path: {filepath}")
            return False
        if not os.path.exists(filepath):
            logger.warning(f"remove_file: file not exist: {filepath}")
            return False
        if not os.path.isfile(filepath):
            logger.warning(f"remove_file: not a file: {filepath}")
            return False
        os.remove(filepath)
        # logger.info(f"remove_file: deleted file: {filepath}")
        return True
    except Exception as exc:
        logger.error(f"remove_file: error when removing file {filepath}: {exc}")
        return False

def empty_folder(dir_name: str, extension: Optional[Optional[str]] = None):
    """Remove all files in given folder.

    Args:
            folder_path (str): Path to ,folder to empty.
    """
    dir_path = Path(dir_name)
    if not dir_path.is_dir():
        print(f"{dir_name} is not a valid directory or does not exist.")
        return 0
    count = 0
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix == extension:
            try:
                file.unlink()
                count += 1
            except Exception as e:
                print(f"Error removing {file}: {e}")
    # print(f"{count} files deleted with extension {extension} in {dir_name}.")
    return count

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
    list_filter_files: List[Any] = []
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
    """
    Read a file and return its content as a list of strings (lines).

    Args:
        filepath (str): Path to the file.
        max_lines (int, optional): Maximum number of lines to read. Defaults to 0 (read all).

    Returns:
        List[str]: List of lines stripped of whitespace.
    """
    with open(filepath) as file:
        lines: List[Any] = []
        n_line = 0
        for line in file:
            if max_lines > 0 and n_line >= max_lines:
                break
            lines.append(line.strip())
            n_line += 1
        return lines


def check_valid_files(list_files: List[str]) -> List[str]:
    """
    Check if files in a list exist and return the valid ones.

    Args:
        list_files (List[str]): List of file paths to check.

    Returns:
        List[str]: List of files that exist.
    """
    list_valid_files: List[Any] = []
    for file_path in list_files:
        if os.path.isfile(file_path):
            list_valid_files.append(file_path)
    return list_valid_files


def load_config_file(path: str) -> Dict[Any, Any]:
    """
    Load a YAML configuration file.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(path) as f:
        result = yaml.safe_load(f)
        return result if result is not None else {}


class FileCacheManager:
    """Manages a cache of files with automatic cleanup when limit is reached."""

    def __init__(self, max_files: int):
        self.max_files = max_files
        self.cache: OrderedDict[str, bool] = OrderedDict()

    def add(self, filepath: str):
        """Add a file to the cache, removing oldest if cache is full."""
        if filepath in self.cache:
            self.cache.move_to_end(filepath)
        else:
            self.cache[filepath] = True
            if len(self.cache) > self.max_files:
                old_path, _ = self.cache.popitem(last=False)
                remove_file(old_path)

    def __contains__(self, filepath: str):
        """Check if filepath is in the cache."""
        return filepath in self.cache

    def remove(self, filepath: str):
        """Remove a file from the filesystem."""
        remove_file(filepath)

    def clear(self):
        """Clear all files from the cache."""
        for filepath in list(self.cache.keys()):
            remove_file(filepath)
        self.cache.clear()
