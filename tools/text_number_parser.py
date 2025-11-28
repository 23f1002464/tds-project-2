# tools/text_number_parser.py

from langchain_core.tools import tool
import re


@tool
def sum_numbers_from_text(text: str) -> int:
    """
    Extract all integers (including negatives) from the given text
    and return their sum.

    Example:
        "Numbers are 10, -3 and 5" -> 12
    """
    numbers = re.findall(r"-?\d+", text)
    ints = [int(n) for n in numbers]
    return sum(ints)
