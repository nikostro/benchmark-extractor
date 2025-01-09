import base64
import os
import tempfile
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import requests


def download_file(url: str) -> Path:
    """
    Downloads a file from the given URL and saves it to a temporary directory.

    Args:
        url: The URL of the file to download

    Returns:
        Path object pointing to the downloaded file

    Raises:
        requests.RequestException: If the download fails
        ValueError: If the URL is invalid or missing a filename
    """
    # Validate URL and extract filename
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}")

    filename = os.path.basename(parsed_url.path)
    if not filename:
        raise ValueError(f"Could not extract filename from URL: {url}")

    # Create temporary directory that persists after function ends
    temp_dir = tempfile.mkdtemp()

    # Construct full path for downloaded file
    file_path = Path(temp_dir) / filename

    # Download the file with streaming to handle large files
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Write the file in chunks
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return file_path


def encode_file(file_path: Union[str, Path]) -> str:
    """
    Reads a file and encodes its contents as a base64 string for sending as an API request.

    Args:
        file_path: Path to the file to encode (string or Path object)

    Returns:
        Base64 encoded string of the file contents

    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read due to permissions
        IOError: For other file reading errors
    """
    # Convert string path to Path object if necessary
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if path is a file
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        # Read and encode the file
        with open(path, "rb") as file:
            binary_data = file.read()
            base64_encoded = base64.b64encode(binary_data)
            return base64_encoded.decode("utf-8")

    except PermissionError as e:
        raise PermissionError(f"Permission denied reading file: {file_path}") from e
    except IOError as e:
        raise IOError(f"Error reading file: {file_path}") from e
