import os
from datasets import load_dataset
import pandas as pd

def download_data():
    print("Downloading dataset...")
    dataset = load_dataset("JDhruv14/Bhagavad-Gita_Dataset")
    
    # Convert to pandas dataframe
    df = dataset['train'].to_pandas()
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/bhagavad_gita.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df.head())

if __name__ == "__main__":
    download_data()
