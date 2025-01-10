from datetime import datetime
from io import StringIO

import pandas as pd
from pydantic import HttpUrl, ValidationError

import utils
from constants import benchmark_results_path, benchmark_sources_path, model_sources_path
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


def crawl_model_url(source_url: str) -> None:
    """Crawl model URL and save source_df and benchmark_results_df"""
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
        benchmark_df.to_csv(benchmark_results_path, index=False)
        logger.info(
            "Successfully loaded Claude response into df and saved benchmark results"
        )

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
        logger.info(f"Appending benchmark_sources {benchmark_sources_path}")

        try:
            # Try to load existing file
            existing_df = pd.read_csv(benchmark_sources_path)

            # Filter source_df to only include new URLs
            new_rows = source_df[~source_df["url"].isin(existing_df["url"])]

            if len(new_rows) > 0:
                # Append new rows and save
                updated_df = pd.concat([existing_df, new_rows], ignore_index=True)
                updated_df.to_csv(benchmark_sources_path, index=False)
                logger.info(
                    f"Successfully appended {len(new_rows)} new rows to existing file"
                )
            else:
                logger.info("No new URLs to add, existing file unchanged")

        except FileNotFoundError:
            # If file doesn't exist, save the new dataframe
            logger.info(
                f"No existing file found at {benchmark_sources_path}, creating new file"
            )
            source_df.to_csv(benchmark_sources_path, index=False)
            logger.info("Successfully created new file with results")

    except Exception as e:
        # If loading/appending fails for other reasons, overwrite the file
        logger.warning(f"Error while trying to append: {e}. Overwriting file instead")
        source_df.to_csv(benchmark_sources_path, index=False)
        logger.info("Successfully overwrote file with new results")


def main(source_df_path: str):
    """Crawls urls from source_df."""

    full_sources_df = pd.read_csv(source_df_path)

    row_to_crawl = full_sources_df.loc[0]
    url_to_crawl = row_to_crawl["url"]
    url_to_crawl = str(url_to_crawl)  # TEMP: gets rid of type warning

    crawl_model_url(url_to_crawl)

    # Update the crawled row
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    full_sources_df.loc[0, "crawled_timestamp"] = current_time
    full_sources_df.loc[0, "success"] = True
    full_sources_df.loc[0, "cost"] = 0.1

    # Save the updated dataframe
    try:
        logger.info(f"Updating crawl metadata in {source_df_path}")
        full_sources_df.to_csv(source_df_path, index=False)
        logger.info("Successfully successfully updated crawl metadata.")
    except Exception as e:
        logger.error(f"Failed to save updated results to {source_df_path}: {e}")
        raise


if __name__ == "__main__":
    main(model_sources_path)
