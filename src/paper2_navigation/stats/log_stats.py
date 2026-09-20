import re
from paper2_navigation.config import RESULTS_FILE
def log_stats(new_stats):
    # Path to your Overleaf project's results.tex file
    tex_file_path = RESULTS_FILE

    # Load existing lines
    with open(tex_file_path, "r") as f:
        lines = f.readlines()

    # Track which keys were updated
    updated_keys = set()

    # Update in-place if key exists
    for i, line in enumerate(lines):
        for key, val in new_stats.items():
            pattern = rf"\\newcommand{{\\{key}}}{{.*}}"
            if re.match(pattern, line.strip()):
                lines[i] = f"\\newcommand{{\\{key}}}{{{val:.2f}}}\n"
                updated_keys.add(key)

    # Append new entries if not found
    for key, val in new_stats.items():
        if key not in updated_keys:
            lines.append(f"\\newcommand{{\\{key}}}{{{val:.2f}}}\n")

    # Write back to file
    with open(tex_file_path, "w") as f:
        f.writelines(lines)
