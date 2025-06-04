import os
import json
import xml.etree.ElementTree as ET
import pandas as pd
import concurrent.futures
import argparse
import logging
import pickle
import time

from functools import reduce
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI

try:
    import tiktoken
    _have_tiktoken = True
except ImportError:
    _have_tiktoken = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##############################################################################
# File Loaders
##############################################################################

def load_csv(file_path):
    try:
        logging.info(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading CSV: {e}")
        return None

def load_json(file_path):
    try:
        logging.info(f"Loading JSON file: {file_path}")
        try:
            return pd.read_json(file_path)
        except ValueError:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.json_normalize(data)
    except Exception as e:
        logging.error(f"Error loading JSON: {e}")
        return None

def load_xml(file_path):
    try:
        logging.info(f"Loading XML file: {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = [{child.tag: child.text for child in record} for record in root]
        return pd.DataFrame(records)
    except Exception as e:
        logging.error(f"Error loading XML: {e}")
        return None

def load_txt(file_path):
    try:
        logging.info(f"Loading TXT file: {file_path}")
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            logging.warning("Detected single-column TXT. Trying tab delimiter...")
            df = pd.read_csv(file_path, delimiter='\t')
        return df
    except Exception as e:
        logging.error(f"Error loading TXT: {e}")
        return None

def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return load_csv(file_path)
    elif ext == '.json':
        return load_json(file_path)
    elif ext == '.xml':
        return load_xml(file_path)
    elif ext == '.txt':
        return load_txt(file_path)
    else:
        logging.warning(f"Unsupported file: {file_path}")
        return None

##############################################################################
# Merge Logic
##############################################################################

def guess_best_merge_key(dfs):
    all_cols = {}
    for df in dfs:
        for col in df.columns:
            all_cols[col] = all_cols.get(col, 0) + 1
    sorted_cols = sorted(all_cols.items(), key=lambda x: (-x[1], x[0]))
    if sorted_cols and sorted_cols[0][1] >= 2:
        chosen = sorted_cols[0][0]
        logging.info(f"Auto-chosen merge key: {chosen}")
        with open("chosen_merge_key.txt", "w") as f:
            f.write(chosen)
        return chosen
    return None

def merge_dataframes_list(dfs, primary_key=None):
    if not dfs:
        return None

    # Print column comparison
    for i, df in enumerate(dfs):
        logging.info(f"DataFrame #{i+1} Columns: {list(df.columns)}")

    if primary_key == "auto":
        primary_key = guess_best_merge_key(dfs)

    if not primary_key:
        logging.warning("No primary key used. Performing vertical concat.")
        return pd.concat(dfs, ignore_index=True, sort=False)

    with_key = [df for df in dfs if primary_key in df.columns]
    without_key = [df for df in dfs if primary_key not in df.columns]

    if not with_key:
        logging.error(f"No DataFrames contain the primary key: {primary_key}")
        return None

    for df in with_key:
        df[primary_key] = df[primary_key].astype(str)

    merged_df = with_key[0]
    for df in with_key[1:]:
        merged_df = pd.merge(merged_df, df, on=primary_key, how='inner')

    if without_key:
        logging.warning(f"{len(without_key)} DataFrames did not have the key. Appending them.")
        merged_df = pd.concat([merged_df] + without_key, ignore_index=True, sort=False)

    return merged_df

##############################################################################
# LangChain Agent Tools
##############################################################################

def data_loaders(input_str: str) -> str:
    try:
        args = dict(item.split("=") for item in input_str.split(",") if "=" in item)
        files_dir = args.get("files_dir", "files")
    except Exception as e:
        return f"Invalid input: {e}"

    if not os.path.exists(files_dir):
        return f"Directory {files_dir} does not exist."

    files = [os.path.join(files_dir, f) for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]
    dataframes = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            if df is not None:
                dataframes.append(df)
                logging.info(f"Loaded {futures[future]} with shape {df.shape}")

    with open("loaded_data.pkl", "wb") as f:
        pickle.dump(dataframes, f)

    return f"Loaded {len(dataframes)} DataFrames from '{files_dir}'"

def merge_dataloaders(input_str: str) -> str:
    try:
        args = dict(item.split("=") for item in input_str.split(",") if "=" in item)
        key = args.get("primary_key", None)
    except Exception as e:
        return f"Invalid input: {e}"

    if not os.path.exists("loaded_data.pkl"):
        return "No loaded data found. Run DataLoaders first."

    with open("loaded_data.pkl", "rb") as f:
        dfs = pickle.load(f)

    merged_df = merge_dataframes_list(dfs, primary_key=key)
    if merged_df is None:
        return "Merging failed."

    with open("merged_data.pkl", "wb") as f:
        pickle.dump(merged_df, f)

    return f"Merged {len(dfs)} DataFrames. Output shape: {merged_df.shape}"

def file_saver(input_str: str) -> str:
    try:
        args = dict(item.split("=") for item in input_str.split(",") if "=" in item)
        output = args.get("output", "merged_output.csv")
    except Exception as e:
        return f"Invalid input: {e}"

    if not os.path.exists("merged_data.pkl"):
        return "No merged data found."

    with open("merged_data.pkl", "rb") as f:
        df = pickle.load(f)

    df.to_csv(output, index=False)
    return f"Merged data saved to {output}"

##############################################################################
# CLI Entry Point
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Merge multiple dataset formats intelligently.")
    parser.add_argument('--files_dir', default='files')
    parser.add_argument('--primary_key', default=None)
    parser.add_argument('--output', default='merged_output.csv')
    args = parser.parse_args()

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        logging.warning("OPENAI_API_KEY not set. LangChain agent may fail.")

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)

    agent = initialize_agent(
        tools=[
            Tool("DataLoaders", data_loaders, "Load files from directory"),
            Tool("MergeDataLoaders", merge_dataloaders, "Merge using key (or auto)"),
            Tool("FileSaver", file_saver, "Save merged CSV")
        ],
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True
    )

    # Run agent commands
    print(agent.run(f"files_dir={args.files_dir}"))
    print(agent.run(f"primary_key={args.primary_key or ''}"))
    print(agent.run(f"output={args.output}"))

    # Show final token estimate
    if _have_tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        print("Example token estimate:", len(enc.encode("merge these files")))

if __name__ == "__main__":
    main()
