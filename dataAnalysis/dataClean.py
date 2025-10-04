import pandas as pd 


def extract_data( inputFilePath: str, outFilePath: str ):
    print('Reading: {inputFilePath}')
    df = pd.read_csv(inputFilePath)
    