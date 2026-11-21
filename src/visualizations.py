"""
Plotly interactive visualization charts engine.
Implements all 12 recommended visualizations with publication-ready styling.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Agronomic color theme
SEVERITY_COLORS = {
    'Low': '#2ca02c',     # Green
    'Medium': '#ff7f0e',  # Amber/Orange
    'High': '#d62728'     # Red
}

CROP_COLORS = ['#2E7D32', '#F57C00', '#1565C0', '#6A1B9A', '#00838F']
CHART_TEMPLATE = 'plotly_white'


def plot_weed_density_by_crop(df: pd.DataFrame):
    """
    Chart 1: Bar chart — Average weed density by crop.
    """
    if df.empty:
        return go.Figure()

    crop_data = df.groupby('crop')['weed_count_per_m2'].mean().reset_index()
    crop_data = crop_data.sort_values('weed_count_per_m2', ascending=False)

    fig = px.bar(
        crop_data,
        x='crop',
        y='weed_count_per_m2',
        color='crop',
        color_discrete_sequence=CROP_COLORS,
        text_auto='.1f',
        title="Average Weed Density by Crop (weeds/m²)",
        labels={'crop': 'Crop Name', 'weed_count_per_m2': 'Avg Weed Count (per m²)'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(showlegend=False, xaxis_title=None, height=380)
    fig.update_traces(textposition='outside')
    return fig


def plot_severity_donut(df: pd.DataFrame):
    """
    Chart 2: Pie/donut chart — Infestation severity distribution.
    """
    if df.empty:
        return go.Figure()

    severity_counts = df['infestation_level'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0).reset_index()
    severity_counts.columns = ['infestation_level', 'count']

    fig = px.pie(
        severity_counts,
        values='count',
        names='infestation_level',
        hole=0.45,
        color='infestation_level',
        color_discrete_map=SEVERITY_COLORS,
        title="Infestation Severity Distribution (%)",
        template=CHART_TEMPLATE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=380, legend_title_text='Severity Level')
    return fig


def plot_yield_loss_by_crop(df: pd.DataFrame):
    """
    Chart 3: Bar chart — Estimated yield loss by crop.
    """
    if df.empty:
        return go.Figure()

    loss_data = df.groupby('crop')['estimated_yield_loss_pct'].mean().reset_index()
    loss_data = loss_data.sort_values('estimated_yield_loss_pct', ascending=False)

    fig = px.bar(
        loss_data,
        x='crop',
        y='estimated_yield_loss_pct',
        color='estimated_yield_loss_pct',
        color_continuous_scale='Reds',
        text_auto='.1f',
        title="Estimated Potential Yield Loss by Crop (%)",
        labels={'crop': 'Crop', 'estimated_yield_loss_pct': 'Avg Yield Loss (%)'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(xaxis_title=None, height=380, coloraxis_showscale=False)
    fig.update_traces(textposition='outside')
    return fig


def plot_severity_by_crop_stacked(df: pd.DataFrame):
    """
    Chart 4: Stacked bar chart — Severity distribution by crop.
    """
    if df.empty:
        return go.Figure()

    grouped = df.groupby(['crop', 'infestation_level']).size().reset_index(name='count')
    # Normalized within crop for percentage
    crop_totals = grouped.groupby('crop')['count'].transform('sum')
    grouped['pct'] = ((grouped['count'] / crop_totals) * 100).round(1)

    fig = px.bar(
        grouped,
        x='crop',
        y='pct',
        color='infestation_level',
        color_discrete_map=SEVERITY_COLORS,
        category_orders={'infestation_level': ['Low', 'Medium', 'High']},
        title="Infestation Severity Breakdown by Crop (%)",
        labels={'crop': 'Crop', 'pct': 'Percentage (%)', 'infestation_level': 'Severity'},
        template=CHART_TEMPLATE,
        text_auto='.1f'
    )
    fig.update_layout(barmode='stack', height=400, xaxis_title=None)
    return fig


def plot_crop_weed_heatmap(df: pd.DataFrame, metric='count'):
    """
    Chart 5: Heatmap — Crop versus weed type.
    metric: 'count' or 'density'
    """
    if df.empty:
        return go.Figure()

    if metric == 'density':
        pivot = df.pivot_table(index='crop', columns='weed_type', values='weed_count_per_m2', aggfunc='mean', fill_value=0).round(1)
        title = "Heatmap: Crop vs. Weed Type (Avg Density weeds/m²)"
        colorbar_title = "Avg Density"
        colorscale = "Viridis"
    else:
        pivot = df.pivot_table(index='crop', columns='weed_type', values='record_id', aggfunc='count', fill_value=0)
        title = "Heatmap: Crop vs. Weed Type (Observation Frequency)"
        colorbar_title = "Occurrences"
        colorscale = "YlGnBu"

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale=colorscale,
        text=pivot.values,
        texttemplate="%{text}",
        textfont={"size": 11},
        hoverongaps=False,
        colorbar=dict(title=colorbar_title)
    ))
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=420,
        xaxis={'tickangle': -25}
    )
    return fig


def plot_infestation_trend(df: pd.DataFrame):
    """
    Chart 6: Line chart — Monthly infestation trend.
    """
    if df.empty:
        return go.Figure()

    monthly = df.groupby(['year_month', 'crop'])['weed_count_per_m2'].mean().reset_index()
    monthly = monthly.sort_values('year_month')

    fig = px.line(
        monthly,
        x='year_month',
        y='weed_count_per_m2',
        color='crop',
        markers=True,
        color_discrete_sequence=CROP_COLORS,
        title="Monthly Weed Infestation Trend (Mean Weeds/m²)",
        labels={'year_month': 'Month', 'weed_count_per_m2': 'Mean Weed Density (per m²)', 'crop': 'Crop'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(height=400, xaxis={'tickangle': -30})
    return fig


def plot_frequent_weeds_bar(df: pd.DataFrame, top_n=8):
    """
    Chart 7: Bar chart — Most frequently observed weed types.
    """
    if df.empty:
        return go.Figure()

    weed_counts = df['weed_type'].value_counts().head(top_n).reset_index()
    weed_counts.columns = ['weed_type', 'count']
    weed_counts = weed_counts.sort_values('count', ascending=True)

    fig = px.bar(
        weed_counts,
        x='count',
        y='weed_type',
        orientation='h',
        color='count',
        color_continuous_scale='Greens',
        text_auto=True,
        title=f"Top {top_n} Most Frequently Observed Weed Species",
        labels={'weed_type': 'Weed Species', 'count': 'Observation Records'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(yaxis_title=None, height=380, coloraxis_showscale=False)
    fig.update_traces(textposition='outside')
    return fig


def plot_weed_density_box(df: pd.DataFrame):
    """
    Chart 8: Box plot — Weed density distribution by crop.
    """
    if df.empty:
        return go.Figure()

    fig = px.box(
        df,
        x='crop',
        y='weed_count_per_m2',
        color='crop',
        color_discrete_sequence=CROP_COLORS,
        points='outliers',
        title="Weed Density Spread and Outliers Across Crops",
        labels={'crop': 'Crop', 'weed_count_per_m2': 'Weeds / m²'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(showlegend=False, height=400, xaxis_title=None)
    return fig


def plot_density_vs_yield_scatter(df: pd.DataFrame):
    """
    Chart 9: Scatter plot — Weed density versus estimated yield loss.
    """
    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df,
        x='weed_count_per_m2',
        y='estimated_yield_loss_pct',
        color='infestation_level',
        color_discrete_map=SEVERITY_COLORS,
        size='estimated_area_affected_pct',
        hover_data=['crop', 'crop_growth_stage', 'treatment_applied'],
        title="Weed Density vs. Estimated Potential Yield Loss (%)",
        labels={
            'weed_count_per_m2': 'Weed Count (per m²)',
            'estimated_yield_loss_pct': 'Estimated Yield Loss (%)',
            'infestation_level': 'Severity'
        },
        template=CHART_TEMPLATE
    )
    # Add trendline manually or simple correlation line
    if len(df) > 3:
        x_vals = np.array([df['weed_count_per_m2'].min(), df['weed_count_per_m2'].max()])
        slope, intercept = np.polyfit(df['weed_count_per_m2'], df['estimated_yield_loss_pct'], 1)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=slope * x_vals + intercept,
            mode='lines',
            name=f'Trendline (slope={slope:.2f})',
            line=dict(color='#333333', dash='dash')
        ))
    fig.update_layout(height=420)
    return fig


def plot_treatment_comparison(df: pd.DataFrame):
    """
    Chart 10: Bar chart — Observations and avg weed density by treatment method.
    """
    if df.empty:
        return go.Figure()

    t_data = df.groupby('treatment_applied').agg(
        record_count=('record_id', 'count'),
        avg_density=('weed_count_per_m2', 'mean')
    ).reset_index().sort_values('avg_density')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=t_data['treatment_applied'],
        y=t_data['avg_density'],
        name='Avg Weed Density (weeds/m²)',
        marker_color='#1E88E5',
        text=t_data['avg_density'].round(1),
        textposition='outside'
    ))
    fig.update_layout(
        title="Comparative Weed Density by Weed Management Treatment",
        template=CHART_TEMPLATE,
        yaxis_title="Avg Weed Density (weeds/m²)",
        height=380,
        xaxis_title=None
    )
    return fig


def plot_growth_stage_infestation(df: pd.DataFrame):
    """
    Chart 11: Bar chart — Infestation by crop growth stage.
    """
    if df.empty:
        return go.Figure()

    stage_order = ['Seedling', 'Vegetative', 'Flowering', 'Maturity']
    stage_data = df.groupby(['crop_growth_stage', 'infestation_level']).size().reset_index(name='count')

    fig = px.bar(
        stage_data,
        x='crop_growth_stage',
        y='count',
        color='infestation_level',
        color_discrete_map=SEVERITY_COLORS,
        category_orders={'crop_growth_stage': stage_order, 'infestation_level': ['Low', 'Medium', 'High']},
        title="Weed Infestation Breakdown by Crop Growth Stage",
        labels={'crop_growth_stage': 'Growth Stage', 'count': 'Observation Records', 'infestation_level': 'Severity'},
        template=CHART_TEMPLATE
    )
    fig.update_layout(barmode='group', height=400, xaxis_title=None)
    return fig


def plot_field_comparison(df: pd.DataFrame):
    """
    Chart 12: Field comparison chart — Weed density by field ID.
    """
    if df.empty:
        return go.Figure()

    field_data = df.groupby('field_id').agg(
        avg_density=('weed_count_per_m2', 'mean'),
        max_density=('weed_count_per_m2', 'max')
    ).reset_index().sort_values('avg_density', ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=field_data['field_id'],
        y=field_data['avg_density'],
        name='Avg Density (weeds/m²)',
        marker_color='#43A047',
        text=field_data['avg_density'].round(1),
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=field_data['field_id'],
        y=field_data['max_density'],
        name='Peak Density Recorded',
        mode='lines+markers',
        line=dict(color='#E53935', width=2)
    ))
    fig.update_layout(
        title="Field-by-Field Infestation Severity Comparison",
        template=CHART_TEMPLATE,
        yaxis_title="Weeds per m²",
        height=400,
        xaxis_title="Field Identifier",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig
