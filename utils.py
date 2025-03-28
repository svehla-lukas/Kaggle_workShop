import pandas as pd
import json
import re

"""CSV file I/O utilities."""


def load_csv_data(file_name: str) -> pd.DataFrame:
    try:
        # Explicitně načteme CSV soubor
        data = pd.read_csv(file_name, keep_default_na=False, na_values=[])
        print(f"Successfully loaded data from {file_name}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_name}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: File at {file_name} is empty or invalid.")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: There was an issue parsing the file at {file_name}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error loading data from {file_name}: {e}")
        return pd.DataFrame()


""" Prepare data for training and testing."""


import json
import re


def parse_description_txt_to_json(txt_path: str, json_path: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = {}
    current_class = None
    current_description = ""
    current_items = {}
    item_id = 0

    category_pattern = re.compile(r"^(\w+):\s*(.+)$")
    item_pattern = re.compile(r"^\s+(\S+)\s+(.+)$")

    for line in lines:
        line = line.rstrip()

        # New category
        match_category = category_pattern.match(line)
        if match_category:
            # Save the previous category
            if current_class:
                result[current_class] = {
                    "description": current_description,
                    "items": current_items if current_items else None,
                }

            current_class = match_category.group(1)
            current_description = match_category.group(2)
            current_items = {}
            item_id = 0  # reset item ID for the new category
            continue

        # Category items
        match_item = item_pattern.match(line)
        if match_item and current_class:
            key = match_item.group(1)
            value = match_item.group(2)
            current_items[key] = {"id": item_id, "description": value}
            item_id += 1

    # Save the last category
    if current_class:
        result[current_class] = {
            "description": current_description,
            "items": current_items if current_items else None,
        }

    # Write to JSON file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Done! JSON has been saved to: {json_path}")
