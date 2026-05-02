"""Runtime configuration for the refactored PackCNN package."""

import os
import sys

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(package_root)

# The original single-file script relied on the working directory and module mode
# for these paths. Keep those paths, and add project_root so `python run.py`
# resolves the local EasyFHE/torch package the same way package-mode execution did.
if project_root not in sys.path:
    sys.path.insert(0, project_root)

for path in (
    "/".join(os.getcwd().split("/")[:-2]),
    "/".join(os.getcwd().split("/")[:-3]),
    "/".join(os.getcwd().split("/")[:-1])
):
    if path and path not in sys.path:
        sys.path.append(path)

# Original DATA_DIR assignments, preserving the effective final value.
# os.environ["DATA_DIR"] = os.path.join(project_root, "PackCNN", "data")
os.environ["DATA_DIR"] = "/data/test/data"
os.environ["DATA_DIR"] = "/data/yky/data"

block_num1 = 3
