import os
import pandas as pd
import sys
from datetime import datetime

def parse_shot_info_dynamic(columns):
    """
    Dynamically parse shot information from a list of columns, handling variations in spacing and format.
    This also reformats the date and drops unnecessary columns.
    """
    try:
        if len(columns) < 13:
            raise ValueError("Insufficient number of columns")

        # Helper function to convert strings to float, handling dashes or invalid values
        def to_float(value):
            try:
                return float(value)
            except ValueError:
                return None

        # Parse the shot ID
        shot_id = columns[2]

        # Parse the date and time, reformatting it to YYYY-MM-DD
        shot_date = columns[4]
        shot_time = columns[5]
        try:
            formatted_date = datetime.strptime(shot_date, '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            breakpoint()
            formatted_date = None

        # Parse the shot latitude, longitude, and water depth
        shot_lat = to_float(columns[6])
        shot_lon = to_float(columns[7])
        ship_lat = to_float(columns[9])  # Column 9 is ship latitude now
        ship_lon = to_float(columns[10])  # Column 10 is ship longitude now
        water_depth = to_float(columns[8])

        # Return relevant parsed information, keeping shot ID at the front and moving water depth to the end
        return [shot_id, formatted_date, shot_time, shot_lat, shot_lon, ship_lat, ship_lon, water_depth]

    except (ValueError, IndexError) as e:
        print(f"Error parsing shot info: {e}")
        return None

def detect_and_standardize_log_format(file):
    """
    Detects the format of the shot log file and standardizes the columns.
    This handles variations in format and drops unnecessary columns.
    """
    data = []

    with open(file, 'r') as f:
        for line in f:
            # Skip lines that are too short or don't have the expected number of columns
            if not line.strip() or len(line.strip().split()) < 13:
                continue

            # Skip lines that are headers (starting with '#')
            if line.startswith('#'):
                continue

            columns = line.strip().split(maxsplit=14)
            parsed_info = parse_shot_info_dynamic(columns)

            if parsed_info:
                # Keep the shot ID, date, time, shot lat/lon, ship lat/lon, and water depth
                data.append(parsed_info)

    return pd.DataFrame(data)

def concatenate_logs(directory, output_file):
    """
    Concatenates all shot log files in a directory and standardizes them into one output file.
    """
    # Get all files with .shotlog or .obsip extension
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if os.path.isfile(os.path.join(directory, f)) and (f.endswith('.shotlog') or f.endswith('.obsip'))]

    concatenated_data = []

    for file in all_files:
        print(f"Processing: {file}")
        data = detect_and_standardize_log_format(file)
        concatenated_data.append(data)

    # Concatenate all the DataFrames
    concatenated_df = pd.concat(concatenated_data, ignore_index=True).drop_duplicates()

    # Save the concatenated DataFrame to the output file as space-delimited text
    concatenated_df.to_csv(output_file, index=False, header=False, sep=' ')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Concatenate all shot logs in a directory.")
    parser.add_argument("directory", help="Directory containing shot log files to concatenate")
    parser.add_argument("output_file", help="Output file to save concatenated logs (space-delimited)")
    args = parser.parse_args()

    concatenate_logs(args.directory, args.output_file)
