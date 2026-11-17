import os
import sys

# Keep tests offline and side-effect free.
os.environ["CHECKPOINT_DB"] = ":memory:"
os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logfire  # noqa: E402

logfire.configure(send_to_logfire=False, console=False)
