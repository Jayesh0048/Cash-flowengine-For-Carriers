import pandas as pd
import glob
import os

def merge_and_deduplicate(file_pattern, output_filename, unique_id_column):
    print(f"[*] Processing {output_filename}...")
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print(f"[!] No files found matching {file_pattern}")
        return
        
    df_list = []
    for file in all_files:
        # Prevent the script from reading a previously generated master file
        if output_filename not in file:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                print(f"    - Loaded {file} ({len(df)} rows)")
            except Exception as e:
                print(f"    [!] Error reading {file}: {e}")

    if df_list:
        # Merge all dataframes
        merged_df = pd.concat(df_list, ignore_index=True)
        
        # Drop duplicates. If unique_id_column is provided, use it. Otherwise, check all columns.
        if unique_id_column and unique_id_column in merged_df.columns:
            final_df = merged_df.drop_duplicates(subset=[unique_id_column], keep='last')
        else:
            final_df = merged_df.drop_duplicates(keep='last')
            
        final_df.to_csv(output_filename, index=False)
        print(f"[✓] Saved {len(final_df)} unique records to {output_filename}\n")

if __name__ == "__main__":
    # 1. Merge all Expenses (Use Expense No. or Receipt No. to deduplicate)
    # This matches expenses.csv, expenses (1).csv, account2_expenses.csv, etc.
    merge_and_deduplicate("*expense*.csv", "master_expenses.csv", "Expense No.")

    # 2. Merge all Loads (Use Load No. to deduplicate)
    # Matches all_loads.csv, account2_all_loads.csv, etc.
    merge_and_deduplicate("*load*.csv", "master_loads.csv", "Load No.")

    # 3. Merge all Invoices (Use Load No. or Invoice No. to deduplicate)
    # Matches invoices.csv, account2_invoiced_loads.csv, etc.
    merge_and_deduplicate("*invoice*.csv", "master_invoices.csv", "Load No.")