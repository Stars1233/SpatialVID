import os
import sys
import pandas as pd
import numpy as np  # 注：当前代码未直接使用numpy，但保留导入以维持与npy处理的兼容性

from utils import npy_to_jsonl


def process_csv_for_npy(data_dir, csv_filename):
    """
    Iterate through subdirectory names in the specified CSV file, 
    find corresponding subdirectories, and call npy_to_jsonl to process data.
    
    Args:
        data_dir (str): Root directory of the data
        csv_filename (str): Name of CSV file containing list of subdirectory names to process
    """
    # Validate and construct CSV file path
    csv_path = os.path.join(data_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found - {csv_path}")
        return
    
    try:
        # Read CSV file and extract subdirectory names from the second column
        df = pd.read_csv(csv_path)
        if len(df.columns) < 2:
            print("Error: CSV file must contain at least two columns")
            return
            
        # Get subdirectory names from the second column (index 1)
        subdir_column = df.columns[1]
        subdirs = df[subdir_column].tolist()
        
        # Process each subdirectory
        for subdir in subdirs:
            # Skip empty or invalid entries
            if not subdir or not isinstance(subdir, str):
                print(f"Warning: Invalid subdirectory name - {subdir}")
                continue
                
            # Construct full path to subdirectory
            subdir_path = os.path.join(data_dir, subdir)
            
            # Check if subdirectory exists
            if not os.path.isdir(subdir_path):
                print(f"Warning: Subdirectory does not exist - {subdir_path}")
                continue
            
            # Locate the poses.npy file in reconstructions subdirectory
            npy_file_path = os.path.join(subdir_path, 'reconstructions', 'poses.npy')
            
            # Verify npy file exists
            if not os.path.isfile(npy_file_path):
                print(f"Warning: Could not find reconstructions/poses.npy in {subdir_path}")
                continue
            
            # Set up target output directory
            output_dir = os.path.join(subdir_path, 'other_data')
            os.makedirs(output_dir, exist_ok=True)  # Create if not exists
            
            # Process the npy file
            print(f"Processing file: {npy_file_path}")
            print(f"Output directory: {output_dir}")
            npy_to_jsonl(npy_file_path, output_dir)
        
        print("Processing completed!")
    
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty")
    except pd.errors.ParserError:
        print("Error: Failed to parse CSV file")
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")


if __name__ == "__main__":
    # Validate command line arguments
    if len(sys.argv) != 3:
        print("Usage: python process_npy.py <data_directory> <csv_filename>")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    csv_filename = sys.argv[2]
    
    # Validate data directory exists
    if not os.path.isdir(data_directory):
        print(f"Error: Data directory does not exist - {data_directory}")
        sys.exit(1)
    
    # Execute main processing function
    process_csv_for_npy(data_directory, csv_filename)
