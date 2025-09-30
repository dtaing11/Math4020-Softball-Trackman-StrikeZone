import csv
import pandas as pd 

def extract_2024_data(input_csv_path, output_csv_path):
    print(f"Reading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    df_2024 = df[df['game_date'].dt.year == 2024]

    print(f"Writing {len(df_2024)} rows to: {output_csv_path}")
    df_2024.to_csv(output_csv_path, index=False)

if __name__ == "__main__": 
    input_file = "../datasets/mlb_pitch_data_2020_2024.csv"       
    output_file = "../datasets/pitches_2024.csv"
    extract_2024_data(input_file, output_file)
