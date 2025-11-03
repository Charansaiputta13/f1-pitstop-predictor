import fastf1
import os
import pandas as pd

def load_race_data(year=2024, gp='Monaco'):
    """
    Fetches and caches Formula 1 race data using FastF1.
    """
    cache_dir = os.path.join('data', 'raw')
    os.makedirs(cache_dir, exist_ok=True)   # ✅ auto-create folder if missing

    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, gp, 'R')
    session.load()

    laps = session.laps
    drivers = laps['Driver'].unique().tolist()

    print(f"✅ Loaded {len(laps)} laps from {gp} {year}")
    print(f"Drivers: {drivers}")
    return laps, session

if __name__ == "__main__":
    laps, session = load_race_data()
    print(laps.head())

