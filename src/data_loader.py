"""
Data loading and validation module.
Handles CSV dataset loading, schema verification, data type coercion, and cleaning.
"""
import os
import io
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'record_id',
    'observation_date',
    'crop',
    'field_id',
    'crop_growth_stage',
    'weed_type',
    'weed_count_per_m2',
    'infestation_level',
    'estimated_area_affected_pct',
    'treatment_applied',
    'estimated_yield_loss_pct',
    'model_detection_confidence'
]

NUMERIC_COLUMNS = [
    'weed_count_per_m2',
    'estimated_area_affected_pct',
    'estimated_yield_loss_pct',
    'model_detection_confidence'
]

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'weed_detection_sample_data_500_records.csv')


def validate_schema(df: pd.DataFrame):
    """
    Validates whether the DataFrame contains all required columns and valid data types.
    Returns:
        tuple (is_valid: bool, issues: list[str])
    """
    issues = []
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {', '.join(missing_cols)}")
        return False, issues

    # Check numeric types
    for col in NUMERIC_COLUMNS:
        non_numeric = pd.to_numeric(df[col], errors='coerce').isna().sum()
        if non_numeric > 0:
            issues.append(f"Column '{col}' contains {non_numeric} non-numeric or missing values.")

    # Check date column
    try:
        parsed_dates = pd.to_datetime(df['observation_date'], errors='coerce')
        invalid_dates = parsed_dates.isna().sum()
        if invalid_dates > 0:
            issues.append(f"Column 'observation_date' contains {invalid_dates} invalid date records.")
    except Exception as e:
        issues.append(f"Error parsing 'observation_date': {str(e)}")

    is_valid = len(issues) == 0
    return is_valid, issues


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the dataset:
    - Parses observation_date to datetime
    - Coerces numeric columns to numeric types
    - Standardizes text casing
    - Derives calendar features (Year, Month, YearMonth)
    """
    clean_df = df.copy()

    # Parse date
    clean_df['observation_date'] = pd.to_datetime(clean_df['observation_date'], errors='coerce')
    clean_df = clean_df.dropna(subset=['observation_date'])

    # Numeric coercion
    clean_df['weed_count_per_m2'] = pd.to_numeric(clean_df['weed_count_per_m2'], errors='coerce').fillna(0).astype(int)
    clean_df['estimated_area_affected_pct'] = pd.to_numeric(clean_df['estimated_area_affected_pct'], errors='coerce').fillna(0.0).astype(float)
    clean_df['estimated_yield_loss_pct'] = pd.to_numeric(clean_df['estimated_yield_loss_pct'], errors='coerce').fillna(0.0).astype(float)
    clean_df['model_detection_confidence'] = pd.to_numeric(clean_df['model_detection_confidence'], errors='coerce').fillna(0.85).astype(float)

    # String cleaning
    for col in ['crop', 'field_id', 'crop_growth_stage', 'weed_type', 'infestation_level', 'treatment_applied']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.strip()

    # Calendar features
    clean_df['year'] = clean_df['observation_date'].dt.year
    clean_df['month'] = clean_df['observation_date'].dt.month
    clean_df['month_name'] = clean_df['observation_date'].dt.strftime('%b %Y')
    clean_df['year_month'] = clean_df['observation_date'].dt.to_period('M').astype(str)

    # Ensure infestation level conforms to standard rule if missing or inconsistent
    def classify_severity(count):
        if count < 12:
            return 'Low'
        elif count <= 29:
            return 'Medium'
        else:
            return 'High'

    if 'infestation_level' not in clean_df.columns or clean_df['infestation_level'].isna().any():
        clean_df['infestation_level'] = clean_df['weed_count_per_m2'].apply(classify_severity)

    return clean_df


def load_dataset(source=None) -> tuple[pd.DataFrame, bool, list[str]]:
    """
    Loads dataset from either:
    1. A file path (str)
    2. An uploaded file buffer (io.BytesIO / Streamlit UploadedFile)
    3. Default data file if source is None
    
    Returns:
        tuple: (df: pd.DataFrame, is_valid: bool, issues: list[str])
    """
    try:
        if source is None:
            if not os.path.exists(DEFAULT_DATA_PATH):
                return pd.DataFrame(), False, [f"Default dataset not found at {DEFAULT_DATA_PATH}"]
            df = pd.read_csv(DEFAULT_DATA_PATH)
        elif isinstance(source, (str, os.PathLike)):
            if not os.path.exists(source):
                return pd.DataFrame(), False, [f"File not found: {source}"]
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(source)

        is_valid, issues = validate_schema(df)
        if not is_valid:
            return df, False, issues

        clean_df = clean_data(df)
        return clean_df, True, issues

    except Exception as e:
        return pd.DataFrame(), False, [f"Failed to load dataset: {str(e)}"]


def get_active_dataframe(session_state=None) -> pd.DataFrame:
    """
    Retrieves the currently active DataFrame from Streamlit session_state
    or loads default sample data if not yet initialized.
    """
    if session_state is not None and 'dataset_df' in session_state and session_state['dataset_df'] is not None:
        return session_state['dataset_df']

    df, valid, _ = load_dataset()
    if valid and session_state is not None:
        session_state['dataset_df'] = df
    return df
