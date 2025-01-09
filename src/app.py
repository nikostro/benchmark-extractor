from io import StringIO

import pandas as pd
from pydantic import HttpUrl, ValidationError

import utils
from constants import claude_pdf_url, sources_df_path
from llm import get_completion
from logger import get_logger

logger = get_logger(__name__)


def is_valid_url(s: str) -> bool:
    try:
        HttpUrl(s)
        logger.debug(f"Valid URL found: {s}")
        return True
    except ValidationError:
        logger.debug(f"Invalid URL found: {s}")
        return False


def parse_source_url(source: str) -> str:
    """Extract PDF URL from general link"""
    logger.debug(f"Parsing source URL: {source}")
    if "arxiv.org" in source:
        if "/abs/" in source:
            result = source.replace("/abs/", "/pdf/") + ".pdf"
            logger.debug(f"Converted arXiv abstract URL to PDF: {result}")
            return result
        if "/pdf/" not in source:
            result = source + ".pdf"
            logger.debug(f"Added PDF extension to arXiv URL: {result}")
            return result
    elif "aclanthology.org" in source:
        result = source + ".pdf"
        logger.debug(f"Added PDF extension to ACL URL: {result}")
        return result
    logger.debug(f"No URL transformation needed: {source}")
    return source


def get_sources(
    input_df: pd.DataFrame, source_type: str, origin_url: str
) -> pd.DataFrame:
    """
    Process input DataFrame to create source tracking DataFrame.
    """
    logger.info(f"Processing sources for type: {source_type}")
    logger.debug(f"Input DataFrame shape: {input_df.shape}")

    df = input_df.copy()

    # Check input df schema
    required_cols = ["name", "source"]
    if not all((col in list(df.columns) for col in required_cols)):
        logger.error(f"Missing required columns. Found: {list(df.columns)}")
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Only keep rows with a valid URL
    initial_rows = len(df)
    df = df[df["source"].apply(is_valid_url)]
    filtered_rows = len(df)
    logger.info(f"Filtered {initial_rows - filtered_rows} invalid URLs")

    df = df.replace("-", None)
    df = df.dropna(subset="source")
    final_rows = len(df)
    logger.info(f"Removed {filtered_rows - final_rows} rows with empty sources")

    # Get output df
    now = pd.Timestamp.now()

    try:
        result_df = pd.DataFrame(
            {
                "name": df["name"],
                "url": df["source"].apply(lambda x: parse_source_url(x)),
                "origin_url": origin_url,
                "crawled_timestamp": None,
                "added_timestamp": now,
                "type": source_type,
                "success": None,
                "cost": None,
            }
        )
        logger.info(f"Created result DataFrame with {len(result_df)} rows")
        return result_df.reset_index(drop=True)
    except Exception:
        logger.error("Error creating result DataFrame", exc_info=True)
        raise


def main(source_url: str):
    """Crawl URL and return source_df"""
    logger.info("Starting main script execution")

    logger.info(f"Downloading source file from {source_url}")
    source_fpath = utils.download_file(source_url)
    logger.info(f"Successfully downloaded file from {source_url}")

    try:
        source_file_base64 = utils.encode_file(source_fpath)
        logger.info("Successfully encoded source file")

    except Exception as e:
        logger.error(f"Failed to encode file {source_fpath}: {e}")
        raise

    try:
        completion = get_completion(source_file_base64)
        logger.info("Successfully got Claude response of benchmark table")

    except Exception as e:
        logger.error(f"Failed to get completion: {e}")
        raise

    try:
        benchmark_df = pd.read_csv(StringIO(completion))
        logger.info("Successfully created benchmark DataFrame")

    except Exception as e:
        logger.error(f"Failed to create benchmark DataFrame: {e}")
        raise

    try:
        logger.info("Processing sources")
        source_df = get_sources(benchmark_df, "benchmark", source_url)
        logger.info("Successfully processed sources")

    except Exception as e:
        logger.error(f"Failed to process sources: {e}")
        raise

    try:
        logger.info(f"Saving results to {sources_df_path}")
        source_df.to_csv(sources_df_path, index=False)
        logger.info("Successfully saved results")

    except Exception as e:
        logger.error(f"Failed to save results to {sources_df_path}: {e}")
        raise


if __name__ == "__main__":
    main(claude_pdf_url)
