"""
Analytics and statistical calculations engine for weed monitoring,
crop comparisons, severity risk scoring, and estimated yield loss.
"""
import pandas as pd
import numpy as np


def get_kpi_metrics(df: pd.DataFrame) -> dict:
    """
    Computes key performance indicators for the dashboard overview.
    """
    if df.empty:
        return {
            'total_records': 0,
            'num_crops': 0,
            'total_weeds': 0,
            'high_infestation_count': 0,
            'high_infestation_pct': 0.0,
            'avg_weed_density': 0.0,
            'avg_affected_area': 0.0,
            'avg_yield_loss': 0.0,
            'dominant_weed': 'N/A',
            'most_affected_crop': 'N/A',
            'avg_confidence': 0.0
        }

    total_records = len(df)
    num_crops = df['crop'].nunique()
    total_weeds = int(df['weed_count_per_m2'].sum())
    high_infest = (df['infestation_level'] == 'High').sum()
    high_infest_pct = round((high_infest / total_records) * 100, 1)
    avg_weed_density = round(float(df['weed_count_per_m2'].mean()), 1)
    avg_affected_area = round(float(df['estimated_area_affected_pct'].mean()), 1)
    avg_yield_loss = round(float(df['estimated_yield_loss_pct'].mean()), 1)
    avg_confidence = round(float(df['model_detection_confidence'].mean()), 2)

    dominant_weed = df['weed_type'].value_counts().index[0] if not df['weed_type'].empty else 'N/A'
    
    # Most affected crop by avg weed density
    crop_density = df.groupby('crop')['weed_count_per_m2'].mean()
    most_affected_crop = crop_density.idxmax() if not crop_density.empty else 'N/A'

    return {
        'total_records': total_records,
        'num_crops': num_crops,
        'total_weeds': total_weeds,
        'high_infestation_count': high_infest,
        'high_infestation_pct': high_infest_pct,
        'avg_weed_density': avg_weed_density,
        'avg_affected_area': avg_affected_area,
        'avg_yield_loss': avg_yield_loss,
        'dominant_weed': dominant_weed,
        'most_affected_crop': most_affected_crop,
        'avg_confidence': avg_confidence
    }


def get_crop_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes detailed statistical breakdown for each crop.
    """
    if df.empty:
        return pd.DataFrame()

    def dominant(x):
        return x.mode()[0] if not x.empty else 'N/A'

    summary = df.groupby('crop').agg(
        record_count=('record_id', 'count'),
        avg_weed_density=('weed_count_per_m2', 'mean'),
        median_weed_density=('weed_count_per_m2', 'median'),
        std_weed_density=('weed_count_per_m2', 'std'),
        avg_affected_area_pct=('estimated_area_affected_pct', 'mean'),
        avg_yield_loss_pct=('estimated_yield_loss_pct', 'mean'),
        dominant_weed=('weed_type', dominant)
    ).reset_index()

    # Round numeric columns
    numeric_cols = ['avg_weed_density', 'median_weed_density', 'std_weed_density',
                    'avg_affected_area_pct', 'avg_yield_loss_pct']
    summary[numeric_cols] = summary[numeric_cols].round(1)

    return summary.sort_values('avg_weed_density', ascending=False)


def get_weed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary metrics per weed species across crops.
    """
    if df.empty:
        return pd.DataFrame()

    def dominant(x):
        return x.mode()[0] if not x.empty else 'N/A'

    summary = df.groupby('weed_type').agg(
        occurrence_count=('record_id', 'count'),
        avg_weed_density=('weed_count_per_m2', 'mean'),
        avg_yield_loss_pct=('estimated_yield_loss_pct', 'mean'),
        avg_area_affected_pct=('estimated_area_affected_pct', 'mean'),
        primary_crop=('crop', dominant)
    ).reset_index()

    # Proportions
    total = len(df)
    summary['occurrence_pct'] = ((summary['occurrence_count'] / total) * 100).round(1)

    numeric_cols = ['avg_weed_density', 'avg_yield_loss_pct', 'avg_area_affected_pct']
    summary[numeric_cols] = summary[numeric_cols].round(1)

    return summary.sort_values('occurrence_count', ascending=False)


def get_crop_weed_matrix(df: pd.DataFrame, metric: str = 'count') -> pd.DataFrame:
    """
    Creates a cross-tabulated matrix of crops vs. weed types.
    metric: 'count' or 'density'
    """
    if df.empty:
        return pd.DataFrame()

    if metric == 'density':
        pivot = df.pivot_table(
            index='crop',
            columns='weed_type',
            values='weed_count_per_m2',
            aggfunc='mean',
            fill_value=0
        ).round(1)
    else:
        pivot = df.pivot_table(
            index='crop',
            columns='weed_type',
            values='record_id',
            aggfunc='count',
            fill_value=0
        )
    return pivot


def get_severity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes infestation severity breakdown (Low, Medium, High) overall and by crop.
    """
    if df.empty:
        return pd.DataFrame()

    order = ['Low', 'Medium', 'High']
    summary = df.groupby(['crop', 'infestation_level']).size().unstack(fill_value=0)
    # Ensure all three columns exist
    for col in order:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary[order]
    summary['Total'] = summary.sum(axis=1)
    summary['High_Infestation_Pct'] = ((summary['High'] / summary['Total']) * 100).round(1)
    return summary.reset_index()


def get_treatment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comparative statistics across weed management treatments applied.
    """
    if df.empty:
        return pd.DataFrame()

    summary = df.groupby('treatment_applied').agg(
        record_count=('record_id', 'count'),
        avg_weed_density=('weed_count_per_m2', 'mean'),
        median_density=('weed_count_per_m2', 'median'),
        avg_affected_area_pct=('estimated_area_affected_pct', 'mean'),
        avg_yield_loss_pct=('estimated_yield_loss_pct', 'mean')
    ).reset_index()

    numeric_cols = ['avg_weed_density', 'median_density', 'avg_affected_area_pct', 'avg_yield_loss_pct']
    summary[numeric_cols] = summary[numeric_cols].round(1)

    return summary.sort_values('avg_weed_density')


def get_field_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks fields by weed infestation risk level based on average density and high-severity share.
    """
    if df.empty:
        return pd.DataFrame()

    field_agg = df.groupby('field_id').agg(
        records=('record_id', 'count'),
        avg_weed_density=('weed_count_per_m2', 'mean'),
        max_weed_density=('weed_count_per_m2', 'max'),
        avg_yield_loss_pct=('estimated_yield_loss_pct', 'mean'),
        high_severity_count=('infestation_level', lambda x: (x == 'High').sum())
    ).reset_index()

    field_agg['high_severity_pct'] = ((field_agg['high_severity_count'] / field_agg['records']) * 100).round(1)
    field_agg['avg_weed_density'] = field_agg['avg_weed_density'].round(1)
    field_agg['avg_yield_loss_pct'] = field_agg['avg_yield_loss_pct'].round(1)

    # Risk level definition
    def assign_risk(row):
        if row['avg_weed_density'] >= 28 or row['high_severity_pct'] >= 40:
            return 'High Risk'
        elif row['avg_weed_density'] >= 18 or row['high_severity_pct'] >= 20:
            return 'Moderate Risk'
        else:
            return 'Controlled'

    field_agg['risk_level'] = field_agg.apply(assign_risk, axis=1)

    return field_agg.sort_values(['high_severity_pct', 'avg_weed_density'], ascending=False)
