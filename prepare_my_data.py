import pandas as pd
import os


original_file_name = 'ground_truth_dataset_top40_intent_30example.csv' # Or your file's name
prepared_file_name = 'prepared_data.csv' # This will be the output file

column_mapping = {
    'intent_definition': 'CSS_mapping',
    'Example phrase': 'utterance',
    'l3_intent(master_intent)': 'intent_level_3',
    'l2_intent': 'intent_level_2',
    'l1_intent': 'intent_level_1'
}


input_path = os.path.join('data', original_file_name)
output_path = os.path.join('data', prepared_file_name)
try:
    df = pd.read_csv(input_path)
    df_renamed = df.rename(columns=column_mapping)
    final_columns = list(column_mapping.values())
    df_final = df_renamed[final_columns]
    df_final.to_csv(output_path, index=False)
    print(f" Success! Your data has been prepared and saved to '{output_path}'")
except Exception as e:
    print(f" Error: {e}")