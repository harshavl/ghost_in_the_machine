import pandas as pd

def data_ingestion_service(file_path):
    """Loads and filters data for death overs."""
    df = pd.read_excel(file_path, engine='xlrd', skiprows=0)  # Try skiprows=0 for files without title row
    df.columns = [c.strip() for c in df.columns]  # Remove any leading/trailing spaces in column names
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    print("Loaded columns:", df.columns.tolist())  # Debug: Print columns to verify 'Phase' is there
    df_death = df[df['Phase'] == 'Death'].copy()
    print(f"Data ingested: {df_death.shape[0]} death over deliveries.")
    return df_death