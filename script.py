import os
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

file_path = os.path.join(project_root, "torch", "fhe", "context.py")
herpn_path = os.path.join(project_root, "examples", "resnet", "gen_weights", "HerPN.py")

backup_path = file_path + ".bak"
shutil.copyfile(file_path, backup_path)

start_line = 409
end_line = 413

start_line2 = 156
end_line2 = 158

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i in range(start_line - 1, end_line):
    if i < 0 or i >= len(lines):
        continue

    if not lines[i].lstrip().startswith("#"):
        lines[i] = "# " + lines[i]

for i in range(start_line2 - 1, end_line2):
    if i < 0 or i >= len(lines):
        continue

    if not lines[i].lstrip().startswith("#"):
        lines[i] = "# " + lines[i]

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(lines)


if not os.path.exists(herpn_path):
    raise FileNotFoundError(f"HerPN.py not found: {herpn_path}")

backup_path = herpn_path + ".bak"
shutil.copyfile(herpn_path, backup_path)

with open(herpn_path, "r", encoding="utf-8") as f:
    content = f.read()

old_line = "model_path = './ResNet20_Aespa.pth'"

indent = " " * 4   

new_block = (
    "import os\n"
    + indent + "project_root = os.path.dirname(os.path.abspath(__file__))\n"
    + indent + 'model_path = os.path.join(project_root, "ResNet20_Aespa.pth")\n'
)

if old_line not in content:
    raise ValueError(f"[ERROR] Target line not found in {herpn_path}: {old_line}")

content_new = content.replace(old_line, new_block)

with open(herpn_path, "w", encoding="utf-8") as f:
    f.write(content_new)

