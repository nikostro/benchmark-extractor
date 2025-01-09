from anthropic import Anthropic

from constants import ANTHROPIC_API_KEY, MODEL_NAME
from logger import get_logger

logger = get_logger(__name__)

prompt = """Please extract the main benchmark table from this PDF document. 
            - Include the name of the benchmarks as rows and the names of the models evaluated as columns.
            - Present the table data in a clear tabular format
            - The columns should be: | name (of benchmark) | test_type (e.g. 5-shot) | models (each column gets its own column) | source |
            - The `source` column should contain the url of the benchmark from the references. If no url is provided in the paper, please provide the the title of the paper which the benchmark references. If this is also unavailable, put a dash (–).
            
            Please maintain the exact numerical values from the original table. Respond with the table and nothing else, in a csv format."""

logger.info("Initialized script with empty PDF base64 string")


logger.debug(f"Constructed messages with prompt length: {len(prompt)}")


anthropic_client = Anthropic(
    default_headers={"anthropic-beta": "pdfs-2024-09-25"}, api_key=ANTHROPIC_API_KEY
)


def get_completion(
    source_base64: str,
    client: Anthropic = anthropic_client,
):
    """Returns extracted table of benchmarks (rows) by model (columns) in csv string"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": source_base64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.info("Requesting completion from Anthropic API")
    try:
        response = client.messages.create(
            model=MODEL_NAME, max_tokens=2048, messages=messages
        )
        logger.info("Successfully received completion")
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error getting completion: {str(e)}", exc_info=True)
        raise
