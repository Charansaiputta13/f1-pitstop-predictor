import pandas as pd
import numpy as np
import os

def load_cleaned_data(path='../data/processed/laps_cleaned.csv'):
    """Loads cleaned lap data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found! Run EDA notebook first.")
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate ML-ready features for pit stop prediction.
    """
    # Sort for consistent per-driver lap sequencing
    df = df.sort_values(['Driver', 'LapNumber']).reset_index(drop=True)

    # Encode tire compound
    compound_map = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2, 'INTERMEDIATE': 3, 'WET': 4}
    df['CompoundCode'] = df['Compound'].map(compound_map).fillna(5)

    # Calculate lap-to-lap deltas
    df['LapDelta'] = df.groupby('Driver')['LapTimeSeconds'].diff()

    # Rolling average lap time per stint (performance trend)
    df['AvgLapInStint'] = (
        df.groupby(['Driver', 'Stint'])['LapTimeSeconds']
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Tire wear proxy: current lap time - stint’s first lap time
    df['StintStartLapTime'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].transform('first')
    df['TireDegradation'] = df['LapTimeSeconds'] - df['StintStartLapTime']

    # Label: whether a driver pits on the next lap
    df['will_pit_next_lap'] = df.groupby('Driver')['Stint'].diff().shift(-1)
    df['will_pit_next_lap'] = df['will_pit_next_lap'].apply(lambda x: 1 if x == 1 else 0)

    # Remove invalid rows
    df = df.dropna(subset=['LapDelta', 'AvgLapInStint'])

    # Feature columns for ML
    feature_cols = [
        'LapNumber',
        'Stint',
        'CompoundCode',
        'LapDelta',
        'AvgLapInStint',
        'TireDegradation',
    ]
    target_col = 'will_pit_next_lap'

    X = df[feature_cols]
    y = df[target_col]

    print(f"✅ Generated {len(X)} samples with {len(feature_cols)} features.")
    return X, y, df


def save_features(X, y, df, output_path='../data/processed/features.csv'):
    """Save combined features + target for modeling."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out = pd.concat([X, y], axis=1)
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved processed dataset to {output_path}")


if __name__ == "__main__":
    df_clean = load_cleaned_data()
    X, y, df_full = engineer_features(df_clean)
    save_features(X, y, df_full)
