import re

counter_index_API = 0

def get_next_api_key(api_keys):
    global counter_index_API
    counter_index_API = counter_index_API + 1
    key = api_keys[(counter_index_API + 1) % len(api_keys)]
    return key

def safe_title(title: str, max_len: int = 100) -> str:
    """
    Sanitize a string to create a safe title by removing special characters 
    and trimming it to a maximum length.

    Args:
        title (str): The original title string.
        max_len (int, optional): The maximum allowed length of the sanitized title. Default is 100.

    Returns:
        str: A cleaned and trimmed version of the title, containing only 
             alphanumeric characters, underscores, hyphens, and spaces.
    """
    # Remove characters not in the allowed set: letters, numbers, underscores, hyphens, and spaces
    cleaned_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title)
    
    # Truncate to the specified max_len and strip leading/trailing spaces
    return cleaned_title[:max_len].strip()


