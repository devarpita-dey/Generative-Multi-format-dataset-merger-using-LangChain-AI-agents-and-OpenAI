
"""
Merge different structured datasets (JSON, XML, CSV, TXT, etc.) into a single CSV file using LangChain AI agents.
If a primary key is provided (e.g., 'id') and at least one DataFrame contains that key, those DataFrames are merged using inner join on that key.
DataFrames that do not contain that key are concatenated side-by-side.
If no DataFrame contains the key at all, or if no primary key is given, all DataFrames are concatenated vertically.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
import pandas as pd
import concurrent.futures
import argparse
import logging
from functools import reduce

# Import LangChain agent classes and OpenAI LLM wrapper.
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

# Set up logging to display info and error messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_openapi_key():
    """Retrieve the OpenAPI key from the environment variable."""
    return os.environ.get("OPENAPI_KEY", None)

def load_csv(file_path):
    """Load a CSV file into a DataFrame."""
    try:
        logging.info(f"Loading CSV file: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading CSV file {file_path}: {e}")
        return None

def load_json(file_path):
    """Load a JSON file into a DataFrame."""
    try:
        logging.info(f"Loading JSON file: {file_path}")
        try:
            # First try direct pd.read_json
            df = pd.read_json(file_path)
        except ValueError:
            # If it fails, try a more general approach
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
        return df
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}")
        return None

def load_xml(file_path):
    """Load an XML file into a DataFrame. Assumes each child of the root is a record."""
    try:
        logging.info(f"Loading XML file: {file_path}")
        tree = ET.parse(file_path)
        root = tree.getroot()
        records = []
        for child in root:
            record = {}
            for subchild in child:
                record[subchild.tag] = subchild.text
            records.append(record)
        return pd.DataFrame(records)
    except Exception as e:
        logging.error(f"Error loading XML file {file_path}: {e}")
        return None

def load_txt(file_path):
    """
    Load a TXT file into a DataFrame.
    This tries comma-delimited first; if that yields only 1 column,
    it retries using tab-delimited.
    """
    try:
        logging.info(f"Loading TXT file: {file_path}")
        df = pd.read_csv(file_path)
        # If it read as a single column, try again with tab delimiter
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, delimiter='\t')
        return df
    except Exception as e:
        logging.error(f"Error loading TXT file {file_path}: {e}")
        return None

def process_file(file_path):
    """Process a file by detecting its extension and using the appropriate loader."""
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
        logging.warning(f"Unsupported file type: {file_path}")
        return None

def merge_dataframes(dfs, primary_key=None):
    """
    Merge a list of DataFrames.
    If primary_key is provided and at least one DataFrame contains that key, merge those DataFrames using inner join.
    DataFrames without that key are concatenated side-by-side (axis=1).
    If no DataFrame contains the primary key (or no primary_key is given), concatenate everything vertically (row-wise).
    """
    if not primary_key:
        # No primary key given -> vertical concat
        return pd.concat(dfs, ignore_index=True, sort=False)

    # Check which DataFrames contain the primary key
    valid_dfs = [df for df in dfs if primary_key in df.columns]
    if not valid_dfs:
        # If none of the DataFrames has the primary key, fall back to vertical concat
        logging.warning(
            f"No DataFrame contains the primary key '{primary_key}'. "
            "Falling back to a vertical concatenation."
        )
        return pd.concat(dfs, ignore_index=True, sort=False)

    # Merge the ones that have the primary key
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=primary_key, how='inner'),
        valid_dfs
    )

    # For DataFrames that do not have the key, concatenate them side by side
    remaining_dfs = [df for df in dfs if df not in valid_dfs]
    for df in remaining_dfs:
        merged_df = pd.concat([merged_df, df], axis=1)

    return merged_df

def merge_datasets_actual(primary_key: str, files_dir: str, output: str) -> str:
    """
    Actual merging function that performs the dataset merging and saves the merged CSV.
    """
    if not os.path.exists(files_dir):
        msg = f"Directory {files_dir} does not exist."
        logging.error(msg)
        return msg

    files = [
        os.path.join(files_dir, f) for f in os.listdir(files_dir)
        if os.path.isfile(os.path.join(files_dir, f))
    ]
    if not files:
        msg = f"No files found in directory {files_dir}."
        logging.error(msg)
        return msg

    logging.info(f"Found {len(files)} files in directory {files_dir}.")

    dataframes = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in files}
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                df = future.result()
                if df is not None:
                    logging.info(f"Processed file: {file_path} with shape {df.shape}")
                    dataframes.append(df)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")

    if not dataframes:
        msg = "No DataFrames were loaded successfully. Exiting."
        logging.error(msg)
        return msg

    merged_df = merge_dataframes(dataframes, primary_key=primary_key)
    if merged_df is None:
        msg = "Merging failed."
        logging.error(msg)
        return msg

    logging.info(f"Merged DataFrame shape: {merged_df.shape}")
    merged_df.to_csv(output, index=False)
    msg = f"Merged CSV saved to {output}"
    logging.info(msg)
    return msg

def merge_datasets_wrapper(input_str: str) -> str:
    """
    Wrapper for the merge_datasets_actual function.
    Expects input string in the format:
      "primary_key=<key>, files_dir=<dir>, output=<file>"
    Parses the input and calls the actual merging function.
    """
    try:
        items = [item.strip() for item in input_str.split(",")]
        args = {}
        for item in items:
            if "=" in item:
                key, value = item.split("=", 1)
                args[key.strip()] = value.strip()
        primary_key = args.get("primary_key", None)
        files_dir = args.get("files_dir", "files")
        output = args.get("output", "merged_output.csv")
    except Exception as e:
        return f"Failed to parse input parameters: {e}"
    return merge_datasets_actual(primary_key, files_dir, output)

def main():
    parser = argparse.ArgumentParser(
        description="Merge various dataset formats into a single CSV file using LangChain AI agents."
    )
    parser.add_argument('--primary_key', type=str, default=None, help='Primary key column to merge on (e.g., id)')
    parser.add_argument('--output', type=str, default='merged_output.csv', help='Output CSV file name')
    parser.add_argument('--files_dir', type=str, default='files', help='Directory containing files to process')
    args = parser.parse_args()

    # Load the OpenAPI key.
    openapi_key = get_openapi_key()
    if not openapi_key:
        logging.warning("OpenAPI key not found. Please set the OPENAPI_KEY environment variable.")

    # Create an OpenAI LLM instance using the OpenAPI key.
    llm = OpenAI(api_key=openapi_key, temperature=0)

    # Define the merge tool as a LangChain Tool.
    merge_tool = Tool(
        name="MergeDatasets",
        func=merge_datasets_wrapper,
        description=(
            "Merges structured datasets from a directory into a single CSV file. "
            "Input should be like: 'primary_key=id, files_dir=files, output=final_merged.csv'"
        )
    )

    # Initialize the LangChain agent with the merge tool.
    agent = initialize_agent([merge_tool], llm, agent="zero-shot-react-description", verbose=True)

    # Construct the command prompt for the agent.
    command = f"primary_key={args.primary_key}, files_dir={args.files_dir}, output={args.output}"
    result = agent.run(command)
    print(result)

if __name__ == '__main__':
    main()
