

import sys
from my_merger import merge_main

# Pre-fill CLI arguments
sys.argv = [
    "my_script.py",
    "--files_dir", "./files",
    "--primary_key", "auto",
    "--output", "result.csv"
]

if __name__ == "__main__":
    merge_main()
