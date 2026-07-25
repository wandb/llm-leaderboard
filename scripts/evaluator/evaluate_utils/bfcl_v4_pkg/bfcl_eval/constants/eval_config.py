import os
import tempfile
from pathlib import Path
from bfcl_eval.constants.category_mapping import VERSION_PREFIX

LOCAL_SERVER_PORT = 1053
LOCAL_SERVER_MAX_CONCURRENT_REQUEST = 100

# Price got from Lambda Cloud, 23.92 per hour for 8x H100, on-demand pay as you go total price
# Reference: https://lambda.ai/pricing
H100_X8_PRICE_PER_HOUR = 23.92

# Directory of the installed package
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Keep generated files out of the vendored source tree even when this module is
# imported outside the leaderboard evaluator. Production runs set
# ``BFCL_PROJECT_ROOT`` to their run-specific output directory.
PROJECT_ROOT = Path(
    os.getenv(
        "BFCL_PROJECT_ROOT",
        str(Path(tempfile.gettempdir()) / "nejumi_bfcl_v4"),
    )
)


RESULT_PATH = PROJECT_ROOT / "result"
SCORE_PATH = PROJECT_ROOT / "score"
DOTENV_PATH = PROJECT_ROOT / ".env"
TEST_IDS_TO_GENERATE_PATH = PROJECT_ROOT / "test_case_ids_to_generate.json"
# Directory that stores all lock files (kept out of the results tree)
LOCK_DIR = PROJECT_ROOT / ".file_locks"

PROMPT_PATH = Path(
    os.getenv("BFCL_DATA_ROOT", str(PACKAGE_ROOT / "data"))
).expanduser().resolve()
MULTI_TURN_FUNC_DOC_PATH = PROMPT_PATH / "multi_turn_func_doc"
POSSIBLE_ANSWER_PATH = PROMPT_PATH / "possible_answer"
MEMORY_PREREQ_CONVERSATION_PATH = PROMPT_PATH / "memory_prereq_conversation"
UTILS_PATH = PACKAGE_ROOT / "scripts"
FORMAT_SENSITIVITY_IDS_PATH = PROMPT_PATH / f"{VERSION_PREFIX}_format_sensitivity.json"

RESULT_FILE_PATTERN = f"{VERSION_PREFIX}_*_result.json"

RED_FONT = "\033[91m"
RESET = "\033[0m"

RESULT_PATH.mkdir(parents=True, exist_ok=True)
SCORE_PATH.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)
