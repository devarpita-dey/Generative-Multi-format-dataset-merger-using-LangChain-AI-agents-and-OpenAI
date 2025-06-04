# Generative-Multi-format-dataset-merger-using-LangChain-AI-agents-and-OpenAI
This repository contains merge structured datasets in different formats (CSV, JSON, XML, TXT) using intelligent LangChain agents powered by OpenAI.

## Features

- ✅ Supports multiple formats: `.csv`, `.json`, `.xml`, `.txt`
- 🤖 LangChain agent-based tool execution pipeline
- 🧠 Auto-detects best primary key for merging (fallback to vertical concat)
- 💾 Intermediate outputs (Pickle) for recovery & performance
- 📊 Final merged output written as `merged_output.csv`

---

## Directory Structure

ai-dataset-merger
 ┣ 📁 files/                    # Folder to put all input CSV, JSON, XML, TXT files
 ┣ 📁 utils/                    # Optional: For splitting file loaders & merging logic
 ┣ 📜 main.py                   # Entry point: CLI tool powered by LangChain + OpenAI
 ┣ 📜 agent_pipeline.py         # Modular pipeline using LangChain Tools and Pickle
 ┣ 📜 requirements.txt          # Dependencies for Python virtual environment
 ┣ 📜 README.md                 # Documentation (see below)
 ┣ 📜 .env                      # (Optional) Store OPENAI_API_KEY here
 ┣ 📜 chosen_merge_key.txt      # Output of auto-detected merge key
 ┣ 📜 loaded_data.pkl           # Intermediate serialized DataFrames
 ┣ 📜 merged_data.pkl           # Serialized final merged DataFrame
 ┗ 📜 merged_output.csv         # Final output CSV (written by file_saver)

**Behind the scenes:**

Step 1: DataLoaders tool loads all files and serializes DataFrames into loaded_data.pkl

Step 2: MergeDataLoaders auto-selects the best primary_key (if auto) and merges datasets

Step 3: FileSaver writes the final CSV to disk (merged_output.csv)

