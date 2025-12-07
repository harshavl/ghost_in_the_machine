import pandas as pd
import numpy as np

def feature_engineering_service(df_death):
    """Engineers pressure and control features."""
    df_death.sort_values(['Match_ID', 'Bowler', 'Over', 'Ball'], inplace=True)
    df_death['previous_runs'] = df_death.groupby(['Match_ID', 'Bowler', 'Over'])['Runs_Conceded'].shift(1)
    df_death['pressure'] = (df_death['previous_runs'] == 0).astype(int)
    df_death['pressure'] = df_death['pressure'].fillna(0)
    
    df_death['bowler_b'] = (df_death['Bowler'] == 'Bowler B').astype(int)
    df_death['interaction'] = df_death['pressure'] * df_death['bowler_b']
    df_death['batter_avg_norm'] = (df_death['Batter_Avg'] - df_death['Batter_Avg'].mean()) / df_death['Batter_Avg'].std()
    df_death['pitch_bowling'] = (df_death['Pitch_Type'] == 'Bowling').astype(int)
    df_death['pitch_neutral'] = (df_death['Pitch_Type'] == 'Neutral').astype(int)
    
    X = np.column_stack([
        np.ones(len(df_death)), df_death['pressure'], df_death['bowler_b'],
        df_death['interaction'], df_death['batter_avg_norm'],
        df_death['pitch_bowling'], df_death['pitch_neutral']
    ])
    y = df_death['Is_Wicket'].values
    print("Features engineered.")
    return X, y, df_death