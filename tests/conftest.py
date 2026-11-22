import os
import sys

# Keep tests offline and side-effect free.
os.environ["CHECKPOINT_DB"] = ":memory:"
os.environ["FEEDBACK_DB"] = ":memory:"
os.environ["HYBRID_SEARCH"] = "false"          # graph tests fake vector search only
os.environ["LTR_MODEL_PATH"] = "/nonexistent/ltr_model.json"
os.environ["QDRANT_CLUSTER_ENDPOINT"] = ""     # never reach a real Qdrant from tests
os.environ["QDRANT_PATH"] = ""
os.environ["LOGFIRE_SEND_TO_LOGFIRE"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logfire  # noqa: E402

logfire.configure(send_to_logfire=False, console=False)
