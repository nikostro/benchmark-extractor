import base64
from pathlib import Path

import pytest
import responses
from requests.exceptions import RequestException

from utils import download_file, encode_file  # Update with your actual module

# Constants for test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_FILE_PATH = TEST_DATA_DIR / "fw9.pdf"
TEST_FILE_URL = "https://www.irs.gov/pub/irs-pdf/fw9.pdf"


@pytest.fixture(scope="session")
def test_data_dir():
    """Ensure test data directory exists"""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def test_file(test_data_dir):
    """Access the test file and its contents"""
    assert TEST_FILE_PATH.exists(), f"Test file not found at {TEST_FILE_PATH}"
    with open(TEST_FILE_PATH, "rb") as f:
        content = f.read()
    return TEST_FILE_PATH, content


@pytest.fixture
def mock_responses():
    """Setup responses for mocking HTTP requests"""
    with responses.RequestsMock() as rsps:
        yield rsps


class TestDownloadFile:
    @pytest.mark.integration
    def test_real_download(self, test_file):
        """Integration test that performs actual download and compares with test file"""
        test_path, expected_content = test_file

        try:
            downloaded_path = download_file(TEST_FILE_URL)

            # Verify file exists and has correct name
            assert downloaded_path.exists()
            assert downloaded_path.is_file()
            assert downloaded_path.name == TEST_FILE_PATH.name

            # Verify it's a valid PDF and compare key characteristics
            with open(downloaded_path, "rb") as f:
                downloaded_content = f.read()

            # Check it's a PDF
            assert downloaded_content.startswith(b"%PDF-")
            assert downloaded_content.rstrip(b"\n").endswith(b"%%EOF")

            # Check file sizes are within 10% of each other
            downloaded_size = len(downloaded_content)
            expected_size = len(expected_content)
            size_difference_ratio = abs(downloaded_size - expected_size) / expected_size
            assert (
                size_difference_ratio < 0.10
            ), f"File size differs by more than 10%: {size_difference_ratio:.2%}"

        finally:
            # Cleanup temporary download
            if downloaded_path.exists():
                downloaded_path.unlink()
            if downloaded_path.parent.exists():
                downloaded_path.parent.rmdir()

    def test_successful_download(self, mock_responses, test_file):
        """Test successful file download"""
        file_path, expected_content = test_file

        # Mock the response with actual file content
        mock_responses.add(
            responses.GET, TEST_FILE_URL, body=expected_content, status=200
        )

        try:
            downloaded_path = download_file(TEST_FILE_URL)

            # Verify
            assert downloaded_path.exists()
            assert downloaded_path.is_file()
            assert downloaded_path.name == TEST_FILE_PATH.name
            with open(downloaded_path, "rb") as f:
                assert f.read() == expected_content

        finally:
            # Cleanup temporary download
            if downloaded_path.exists():
                downloaded_path.unlink()
            if downloaded_path.parent.exists():
                downloaded_path.parent.rmdir()

    def test_invalid_url(self):
        """Test handling of invalid URL"""
        with pytest.raises(ValueError, match="Invalid URL"):
            download_file("not-a-url")

    def test_missing_filename(self):
        """Test URL without filename"""
        with pytest.raises(ValueError, match="Could not extract filename"):
            download_file("https://example.com/")

    def test_download_failure(self, mock_responses):
        """Test handling of download failure"""
        mock_responses.add(responses.GET, TEST_FILE_URL, status=404)

        with pytest.raises(RequestException):
            download_file(TEST_FILE_URL)


class TestEncodeFile:
    def test_successful_encoding(self, test_file):
        """Test successful file encoding"""
        file_path, content = test_file
        encoded = encode_file(file_path)

        # Decode and verify
        import base64

        decoded = base64.b64decode(encoded.encode("utf-8"))
        assert decoded == content

    def test_nonexistent_file(self):
        """Test handling of nonexistent file"""
        with pytest.raises(FileNotFoundError):
            encode_file(TEST_DATA_DIR / "nonexistent.txt")

    def test_directory_path(self, test_data_dir):
        """Test handling of directory path instead of file"""
        with pytest.raises(ValueError, match="Path is not a file"):
            encode_file(test_data_dir)

    def test_permission_error(self, test_file, monkeypatch):
        """Test handling of permission error"""
        file_path, _ = test_file

        def mock_open(*args, **kwargs):
            raise PermissionError("Permission denied")

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(PermissionError):
            encode_file(file_path)

    @pytest.mark.parametrize("path_type", [str, Path])
    def test_path_types(self, test_file, path_type):
        """Test both string and Path inputs"""
        file_path, content = test_file
        path = path_type(file_path)
        encoded = encode_file(path)
        decoded = base64.b64decode(encoded.encode("utf-8"))
        assert decoded == content
