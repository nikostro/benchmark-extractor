from io import StringIO

import pandas as pd
from anthropic import Anthropic
from pydantic import HttpUrl, ValidationError

from logger import get_logger
from src.constants import ANTHROPIC_API_KEY, MODEL_NAME, claude_pdf_url, source_df_path

logger = get_logger(__name__)

prompt = """Please extract the main benchmark table from this PDF document. 
            - Include the name of the benchmarks as rows and the names of the models evaluated as columns.
            - Present the table data in a clear tabular format
            - The columns should be: | name (of benchmark) | test_type (e.g. 5-shot) | models (each column gets its own column) | source |
            - The `source` column should contain the url of the benchmark from the references. If no url is provided in the paper, please provide the the title of the paper which the benchmark references. If this is also unavailable, put a dash (–).
            
            Please maintain the exact numerical values from the original table. Respond with the table and nothing else, in a csv format."""

pdf_base64_string = None  # this would be the input to Claude

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_base64_string,
                },
            },
            {"type": "text", "text": prompt},
        ],
    }
]


def get_completion(client, messages):
    return (
        client.messages.create(model=MODEL_NAME, max_tokens=2048, messages=messages)
        .content[0]
        .text
    )


client = Anthropic(
    default_headers={"anthropic-beta": "pdfs-2024-09-25"}, api_key=ANTHROPIC_API_KEY
)


def is_valid_url(s: str) -> bool:
    try:
        HttpUrl(s)
        return True
    except ValidationError:
        return False


def parse_source_url(source: str) -> str:
    """Extract PDF URL from general link"""
    if "arxiv.org" in source:
        if "/abs/" in source:
            return source.replace("/abs/", "/pdf/") + ".pdf"
        if "/pdf/" not in source:
            return source + ".pdf"
    elif "aclanthology.org" in source:
        return source + ".pdf"
    return source


def get_sources(
    input_df: pd.DataFrame, source_type: str, origin_url: str
) -> pd.DataFrame:
    """
    Input: df with schema:
        - name: str (name of benchmark or model)
        - source: str
    Output: df with schema:
        - name: str
        - url: str - url of pdf/source to crawl
        - crawled_timestamp: datetime64 - when source was crawled
        - added_timestamp: datetime64 - when source was added to db
        - type: SourceType (model/benchmark)
        - success: bool - whether crawl succeeded
    """
    df = input_df.copy()

    # Check input df schema
    assert all((col in list(df.columns) for col in ["name", "source"]))

    # Only keep rows with a valid URL
    df = df[df["source"].apply(is_valid_url)]

    df = df.replace("-", None)

    df = df.dropna(subset="source")

    # Get output df

    now = pd.Timestamp.now()

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

    return result_df


if __name__ == "__main__":
    completion = """name,test_type,Claude 3 Opus,Claude 3 Sonnet,Claude 3 Haiku,GPT-4,GPT-3.5,Gemini 1.0 Ultra,Gemini 1.5 Pro,Gemini 1.0 Pro,source
    MMLU,5-shot,86.8%,79.0%,75.2%,86.4%,70.0%,83.7%,81.9%,71.8%,https://arxiv.org/abs/2201.11903
    MATH,4-shot,61%,40.5%,40.9%,52.9%,34.1%,53.2%,58.5%,32.6%,https://arxiv.org/abs/2103.03874
    GSM8K,0-shot CoT,95.0%,92.3%,88.9%,92.0%,57.1%,94.4%,91.7%,86.5%,https://arxiv.org/abs/2110.14168
    HumanEval,0-shot,84.9%,73.0%,75.9%,67.0%,48.1%,74.4%,71.9%,67.7%,https://arxiv.org/abs/2107.03374
    GPQA (Diamond),0-shot CoT,50.4%,40.4%,33.3%,35.7%,28.1%,-,-,-,https://arxiv.org/abs/2311.12022
    MGSM,0-shot,90.7%,83.5%,75.1%,74.5%,-,79.0%,88.7%,63.5%,https://arxiv.org/abs/2210.03057
    DROP,F1 Score,83.1,78.9,78.4,80.9,64.1,82.4,78.9,74.1,https://aclanthology.org/N19-1246/
    BIG-Bench-Hard,3-shot CoT,86.8%,82.9%,73.7%,83.1%,66.6%,83.6%,84.0%,75.0%,Beyond the imitation game: Quantifying and extrapolating the capabilities of language models
    ARC-Challenge,25-shot,96.4%,93.2%,89.2%,96.3%,85.2%,-,-,-,Think you have Solved Question Answering? Try ARC the AI2 Reasoning Challenge
    HellaSwag,10-shot,95.4%,89.0%,85.9%,95.3%,85.5%,87.8%,92.5%,84.7%,https://arxiv.org/abs/1905.07830
    PubMedQA,5-shot,75.8%,78.3%,76.0%,74.4%,60.2%,-,-,-,https://arxiv.org/abs/1909.06146
    WinoGrande,5-shot,88.5%,75.1%,74.2%,87.5%,-,-,-,-,https://arxiv.org/abs/1907.10641
    RACE-H,5-shot,92.9%,88.8%,87.0%,-,-,-,-,-,https://aclanthology.org/D17-1082
    APPS,0-shot,70.2%,55.9%,54.8%,-,-,-,-,-,https://arxiv.org/abs/2108.07732
    MBPP,Pass@1,86.4%,79.4%,80.4%,-,-,-,-,-,-"""

    df = pd.read_csv(StringIO(completion))

    source_df = get_sources(df, "benchmark", claude_pdf_url)

    source_df.to_csv(source_df_path, index=False)
