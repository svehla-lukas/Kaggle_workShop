import pandas as pd
import json
import re
import matplotlib.pyplot as plt
import numpy as np

""" support blocks """


def to_numpy(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)


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


def auto_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()

    for column in df_cleaned.columns:
        values = df_cleaned[column].dropna().unique()

        can_be_numeric = True
        for v in values:
            if isinstance(v, str) and v.strip().upper() == "NA":
                continue
            try:
                float(v)
            except:
                can_be_numeric = False
                break

        if can_be_numeric:
            df_cleaned[column] = df_cleaned[column].replace("NA", np.nan)
            df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors="coerce")
            # NaN u čísel zůstane (budeme řešit později imputací nebo ignorováním)
        else:
            df_cleaned[column] = df_cleaned[column].astype(str).str.strip().str.upper()
            df_cleaned[column] = df_cleaned[column].fillna("MISSING")  # důležité!

    return df_cleaned


""" plot data """


def plot_history(
    history_dict: dict, metric: str = "mae"
) -> tuple[list[float], list[float]]:
    """
    Plots training and validation metrics from the history dictionary.

    Args:
        history_dict (dict): The history dictionary, usually from model.fit().history
        metric (str): The metric to plot ("mae" or "loss")

    Returns:
        tuple: (training_values, validation_values)
    """
    if metric not in ["mae", "loss"]:
        raise ValueError("Supported metrics are 'mae' and 'loss'")

    train_metric = history_dict[metric]
    val_metric = history_dict[f"val_{metric}"]
    epochs = range(1, len(train_metric) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metric, "bo-", label=f"Training {metric.upper()}")
    plt.plot(epochs, val_metric, "ro-", label=f"Validation {metric.upper()}")
    plt.title(f"Training and Validation {metric.upper()}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.upper() if metric == "loss" else "Mean Absolute Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # return train_metric, val_metric


def plot_rescaled_history(history_dict, metric: str, scaler):
    # Get scaling factor from StandardScaler
    scale = scaler.scale_[0]

    # Extract metrics from history
    train_metric = history_dict[metric]
    val_metric = history_dict[f"val_{metric}"]
    epochs = range(1, len(train_metric) + 1)

    # Rescale metrics back to original dollar scale
    if metric == "loss":  # MSE → scale^2
        train_metric_rescaled = [m * (scale**2) for m in train_metric]
        val_metric_rescaled = [m * (scale**2) for m in val_metric]
        y_label = "Mean Squared Error ($)"
    else:  # MAE → scale
        train_metric_rescaled = [m * scale for m in train_metric]
        val_metric_rescaled = [m * scale for m in val_metric]
        y_label = "Mean Absolute Error ($)"

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metric_rescaled, "bo-", label=f"Training {metric.upper()}")
    plt.plot(epochs, val_metric_rescaled, "ro-", label=f"Validation {metric.upper()}")
    plt.title(f"Training and Validation {metric.upper()} (Rescaled to USD)")
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
