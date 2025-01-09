from io import StringIO

import pandas as pd

from app import get_sources
from constants import claude_pdf_url

example_completion = """name,test_type,Claude 3 Opus,Claude 3 Sonnet,Claude 3 Haiku,GPT-4,GPT-3.5,Gemini 1.0 Ultra,Gemini 1.5 Pro,Gemini 1.0 Pro,source
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
example_sources_csv = """name,url,origin_url,crawled_timestamp,added_timestamp,type,success,cost
    MMLU,https://arxiv.org/pdf/2201.11903.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    MATH,https://arxiv.org/pdf/2103.03874.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    GSM8K,https://arxiv.org/pdf/2110.14168.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    HumanEval,https://arxiv.org/pdf/2107.03374.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    GPQA (Diamond),https://arxiv.org/pdf/2311.12022.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    MGSM,https://arxiv.org/pdf/2210.03057.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    DROP,https://aclanthology.org/N19-1246/.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    HellaSwag,https://arxiv.org/pdf/1905.07830.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    PubMedQA,https://arxiv.org/pdf/1909.06146.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    WinoGrande,https://arxiv.org/pdf/1907.10641.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    RACE-H,https://aclanthology.org/D17-1082.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
    APPS,https://arxiv.org/pdf/2108.07732.pdf,https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf,,2025-01-09 10:54:17.495720,benchmark,,
"""


def test_get_sources():
    # Get the actual result
    completion_df = pd.read_csv(StringIO(example_completion))
    actual_df = get_sources(
        input_df=completion_df,
        source_type="benchmark",
        origin_url=claude_pdf_url,
    )

    # Convert actual DataFrame to CSV string
    actual_csv = StringIO()
    actual_df.to_csv(actual_csv, index=False)
    actual_csv_str = actual_csv.getvalue()

    # Read back the CSV string to DataFrame
    actual_reread = pd.read_csv(StringIO(actual_csv_str))
    expected_df = pd.read_csv(StringIO(example_sources_csv))

    # Check columns are identical
    assert set(actual_reread.columns) == set(expected_df.columns)

    # Compare everything except added_timestamp
    timestamp_col = "added_timestamp"
    cols_to_compare = [col for col in actual_reread.columns if col != timestamp_col]

    pd.testing.assert_frame_equal(
        actual_reread[cols_to_compare].reset_index(drop=True),
        expected_df[cols_to_compare].reset_index(drop=True),
        check_exact=True,
    )
