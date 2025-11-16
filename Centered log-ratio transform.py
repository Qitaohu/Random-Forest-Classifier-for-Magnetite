"""
Centered Log-Ratio (CLR) Transformation for Geochemical Data

This script performs CLR transformation on geochemical data to handle compositional data constraints.
It replaces zero values with small random numbers to avoid undefined logarithms and then applies CLR transformation.

Author: Qi-Tao Hu
Contact: qitaohu@mail.ustc.edu.cn
GitHub Repository: https://github.com/Qitaohu/Random-Forest-Classifier-for-Magnetite

Usage:
1. Modify the file paths
2. Run the script
"""

import pandas as pd
import numpy as np
import os


def clr_transform(input_file_path, output_file_path=None, random_seed=42):
    """
    Perform Centered Log-Ratio (CLR) transformation on geochemical data.

    Parameters:
    -----------
    input_file_path : str
        Path to the input Excel file containing the raw geochemical data
    output_file_path : str, optional
        Path for the output Excel file. If None, will be generated automatically
    random_seed : int, optional
        Random seed for reproducibility when replacing zero values (default: 42)

    Returns:
    --------
    pandas.DataFrame
        CLR-transformed dataframe
    """

    # Set random seed for reproducible zero-value replacement
    np.random.seed(random_seed)

    # Load dataset
    print(f"Loading data from: {input_file_path}")
    df = pd.read_excel(input_file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Select only numeric columns for transformation
    # Non-numeric columns (e.g., sample IDs, location data) will be preserved
    df_numeric = df.select_dtypes(include=[np.number])
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    print(f"Numeric columns selected for transformation: {len(df_numeric.columns)}")
    print(f"Non-numeric columns preserved: {len(non_numeric_cols)}")

    # Generate random matrix for zero-value replacement
    # Zero values in compositional data cause issues with log transformations
    # We replace them with small random values between 1e-4 and 1e-2
    print("Replacing zero values with small random numbers...")
    random_matrix = np.random.uniform(1e-4, 1e-2, size=df_numeric.shape)
    df_numeric_replaced = df_numeric.mask(df_numeric == 0, other=random_matrix)

    # Count zero replacements for reporting
    zero_count = (df_numeric == 0).sum().sum()
    print(f"Zero values replaced: {zero_count}")

    def clr_transform_row(row):
        """
        Apply CLR transformation to a single row of data.

        CLR transformation: clr(x) = log(x) - mean(log(x))
        This transforms compositional data from simplex space to real space.

        Parameters:
        -----------
        row : pandas.Series
            A row of geochemical data

        Returns:
        --------
        pandas.Series
            CLR-transformed row
        """
        # Calculate geometric mean of the row
        geometric_mean = np.exp(np.mean(np.log(row)))

        # Apply CLR transformation: log(x) - log(geometric_mean)
        clr_transformed = np.log(row) - np.log(geometric_mean)

        return clr_transformed

    # Apply CLR transformation row-wise to all numeric data
    print("Applying CLR transformation...")
    clr_transformed_rows = df_numeric_replaced.apply(clr_transform_row, axis=1)

    # Recombine CLR-transformed numeric data with preserved non-numeric data
    df_final = pd.concat([df[non_numeric_cols], clr_transformed_rows], axis=1)

    # Generate output file path if not provided
    if output_file_path is None:
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        name, ext = os.path.splitext(input_filename)
        output_file_path = os.path.join(input_dir, f"{name}_CLR{ext}")

    # Save the final CLR-transformed dataset
    df_final.to_excel(output_file_path, index=False)
    print(f"CLR-transformed data saved to: {output_file_path}")
    print(f"Final dataset shape: {df_final.shape}")

    return df_final


def main():
    """
    Main function to execute CLR transformation with specified file paths.
    Modify the file paths according to your data location.
    """

    # FILE PATHS - MODIFY THESE ACCORDING TO YOUR DATA LOCATION
    # Input: Raw geochemical data file path
    INPUT_FILE_PATH = r"Training DATA.xlsx"

    # Output: CLR-transformed data file path (optional - will be auto-generated if None)
    OUTPUT_FILE_PATH = r"Training DATA CLR.xlsx"

    # Perform CLR transformation
    clr_result = clr_transform(
        input_file_path=INPUT_FILE_PATH,
        output_file_path=OUTPUT_FILE_PATH,
        random_seed=42  # For reproducible results
    )

    print("CLR transformation completed successfully!")


if __name__ == "__main__":
    main()
