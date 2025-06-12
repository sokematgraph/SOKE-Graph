"""
File:		utils.py
Purpose:	containing all utility functions used in other files
"""
import argparse
from pathlib import Path


def check_file(path: str) -> Path:
    """
    Check an input file exists and is readable
    """
    path = Path(path)
    if path.exists() and path.is_file():
        return path.resolve()
    else:
        raise argparse.ArgumentTypeError(
            f"{path} can't be found, please double check the path"
        )

def validate_range(value_type, minimum, maximum):
    """Determine whether arg value is within minimum and maximum range"""

    def range_checker(arg):
        """
        argparse type function to determine value is within specified range
        """
        if value_type is float:
            try:
                val = float(arg)
            except ValueError:
                raise argparse.ArgumentTypeError("Must be a float")
        elif value_type is int:
            try:
                val = int(arg)
            except ValueError:
                raise argparse.ArgumentTypeError("Must be an int")

        if val < minimum or val > maximum:
            raise argparse.ArgumentTypeError(f"must be in range [{minimum}-{maximum}]")
        return val

    return range_checker


