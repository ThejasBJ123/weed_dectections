"""
Data preprocessing and multi-criteria filtering module.
"""
import pandas as pd
from typing import Optional, List


def filter_dataset(
    df: pd.DataFrame,
    crops: Optional[List[str]] = None,
    date_range: Optional[tuple] = None,
    fields: Optional[List[str]] = None,
    weed_types: Optional[List[str]] = None,
    growth_stages: Optional[List[str]] = None,
    infestation_levels: Optional[List[str]] = None,
    treatments: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Filters a DataFrame based on multiple user-selected criteria.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Filter by Crops
    if crops:
        filtered = filtered[filtered['crop'].isin(crops)]

    # Filter by Date Range
    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if pd.notna(start_date) and pd.notna(end_date):
            filtered = filtered[
                (filtered['observation_date'] >= start_date) &
                (filtered['observation_date'] <= end_date)
            ]

    # Filter by Fields
    if fields:
        filtered = filtered[filtered['field_id'].isin(fields)]

    # Filter by Weed Types
    if weed_types:
        filtered = filtered[filtered['weed_type'].isin(weed_types)]

    # Filter by Crop Growth Stages
    if growth_stages:
        filtered = filtered[filtered['crop_growth_stage'].isin(growth_stages)]

    # Filter by Infestation Levels
    if infestation_levels:
        filtered = filtered[filtered['infestation_level'].isin(infestation_levels)]

    # Filter by Treatments Applied
    if treatments:
        filtered = filtered[filtered['treatment_applied'].isin(treatments)]

    return filtered


def get_filter_options(df: pd.DataFrame) -> dict:
    """
    Extracts sorted unique values for each filterable attribute from the DataFrame.
    """
    if df.empty:
        return {
            'crops': [],
            'fields': [],
            'weed_types': [],
            'growth_stages': [],
            'infestation_levels': ['Low', 'Medium', 'High'],
            'treatments': [],
            'min_date': None,
            'max_date': None
        }

    return {
        'crops': sorted(df['crop'].dropna().unique().tolist()),
        'fields': sorted(df['field_id'].dropna().unique().tolist()),
        'weed_types': sorted(df['weed_type'].dropna().unique().tolist()),
        'growth_stages': sorted(df['crop_growth_stage'].dropna().unique().tolist()),
        'infestation_levels': ['Low', 'Medium', 'High'],
        'treatments': sorted(df['treatment_applied'].dropna().unique().tolist()),
        'min_date': df['observation_date'].min().date() if 'observation_date' in df else None,
        'max_date': df['observation_date'].max().date() if 'observation_date' in df else None
    }
