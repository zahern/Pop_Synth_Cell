import pandas as pd
import os
import sys

class Helper:
    def __init__(self):
        """
        Initialise the helper class
        """
        pass

    def clean_my_datagan(self, csv_file, select_columns=None):
        """
        Reads a CSV file, normalises column names to lowercase, and returns either
        all columns (select_columns=None) or only the requested subset.

        Parameters:
        - csv_file: Path to the CSV file to read.
        - select_columns: List of column names to keep (case-insensitive).
                          If None, all columns are returned.

        Returns:
        - A pandas DataFrame.

        Raises:
        - ValueError: If any of the specified columns are missing from the dataset.
        """
        raw = pd.read_csv(csv_file, dtype=str, low_memory=True)
        my_data_for_comparison = raw.astype(str).apply(
            lambda col: col.str.replace('_', '', regex=False)
        ).apply(pd.to_numeric, errors='coerce')
        # Genuinely non-numeric columns (e.g. the 'source' provenance flag:
        # gan/microdata/infill) coerce to all-NaN above, which silently
        # destroys them in every downstream output (all fitted zone files up
        # to run 23883xxx have an all-NaN 'source'). Keep the original string
        # values for any column that had data but lost all of it to coercion.
        for _c in raw.columns:
            if my_data_for_comparison[_c].isna().all() and raw[_c].notna().any():
                my_data_for_comparison[_c] = raw[_c]
        my_data_for_comparison.columns = my_data_for_comparison.columns.str.lower()

        # Normalize synthetic sentinel codes so they line up with the census tables.
        if "mrerd" in my_data_for_comparison.columns:
            my_data_for_comparison["mrerd"] = my_data_for_comparison["mrerd"].replace({0: 23})
        if "rntrd" in my_data_for_comparison.columns:
            my_data_for_comparison["rntrd"] = my_data_for_comparison["rntrd"].replace({0: 27})

        if select_columns is None:
            return my_data_for_comparison

        select_columns = [col.lower() for col in select_columns]
        missing_columns = set(select_columns) - set(my_data_for_comparison.columns)
        if missing_columns:
            raise ValueError(f"The following columns are missing in the data: {', '.join(missing_columns)}")
        return my_data_for_comparison[select_columns]