import pandas as pd

def extract_2020_to_2024_data(input_csv_path):
    print(f"Reading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')

    for year in range(2020, 2025):  
        df_year = df[df['game_date'].dt.year == year]
        output_file = f"../datasets/pitches_{year}.csv"
        print(f"Writing {len(df_year)} rows to: {output_file}")
        df_year.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "../datasets/mlb_pitch_data_2020_2024.csv"
    extract_2020_to_2024_data(input_file)
