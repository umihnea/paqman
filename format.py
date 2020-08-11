#!/usr/bin/env python3
import os
import subprocess

EXCLUDED_SUBDIRS = [
    "env",
    "data",
    "notebooks",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".idea",
    ".git",
]


def recursively_format_docstrings(node):
    with os.scandir(node) as it:
        for entry in it:
            joined_path = os.path.join(node, entry)
            if entry.name.endswith(".py") and entry.is_file():
                subprocess.run(["docformatter", "-i", joined_path])
            elif entry.is_dir() and entry.name not in EXCLUDED_SUBDIRS:
                recursively_format_docstrings(joined_path)


def format_python():
    subprocess.run(["black", "."])


if __name__ == "__main__":
    format_python()

    root = os.getcwd()
    recursively_format_docstrings(root)
