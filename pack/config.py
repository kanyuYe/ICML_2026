import os
import sys

package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(package_root)


if project_root not in sys.path:
    sys.path.insert(0, project_root)

for path in (
    "/".join(os.getcwd().split("/")[:-2]),
    "/".join(os.getcwd().split("/")[:-3]),
    "/".join(os.getcwd().split("/")[:-1])
):
    if path and path not in sys.path:
        sys.path.append(path)

os.environ["DATA_DIR"] = os.path.join(project_root, "PackCNN", "data")
# os.environ["DATA_DIR"] = "/data/test/data"

block_num1 = 3
