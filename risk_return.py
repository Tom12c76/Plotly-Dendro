import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import calculate_annualized_metrics

def calculate_cluster_metrics(returns_df, cluster_dict):
    """
    Calculate risk-return metrics for each fund and add cluster information
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    cluster_dict : dict
        Dictionary mapping fund IDs to cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with risk-return metrics and cluster labels for each fund
    """
    results = []
    
    for fund, cluster in cluster_dict.items():
        fund_returns = returns_df[fund]
        
        # Calculate annualized return and volatility
        ann_return = fund_returns.mean() * 252  # Assuming 252 trading days
        ann_volatility = fund_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_volatility if ann_volatility > 0 else 0
        
        results.append({
            'Fund': fund,
            'Cluster': cluster,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_volatility,
            'Sharpe Ratio': sharpe
        })
    
    return pd.DataFrame(results)

def create_risk_return_scatter(metrics_df, fund_names=None, size_metric='Sharpe Ratio'):
    """
    Create a risk-return scatter plot with cluster coloring
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with risk-return metrics and cluster information
    fund_names : dict, optional
        Dictionary mapping fund IDs to display names
    size_metric : str, optional
        Metric to use for point sizing, by default 'Sharpe Ratio'
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Make a copy to avoid modifying original
    plot_df = metrics_df.copy()
    
    # Convert size metric to absolute values and add a small constant for display
    if size_metric in plot_df.columns:
        size_values = np.abs(plot_df[size_metric].values) + 1
    else:
        size_values = 5  # Default size if metric not found
    
    # Add fund names if provided
    if fund_names:
        plot_df['Fund Name'] = plot_df['Fund'].map(
            lambda x: fund_names.get(x, x)
        )
    else:
        plot_df['Fund Name'] = plot_df['Fund']
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Annualized Volatility',
        y='Annualized Return',
        color='Cluster',
        hover_name='Fund Name',
        text='Fund Name',
        size=size_values,
        size_max=20,
        title="Risk-Return Analysis by Cluster"
    )
    
    # Add efficient frontier line (simple illustration)
    min_vol = plot_df['Annualized Volatility'].min() * 0.9
    max_vol = plot_df['Annualized Volatility'].max() * 1.1
    
    vol_range = np.linspace(min_vol, max_vol, 100)
    frontier_return = 0.1 * np.sqrt(vol_range) + 0.01  # Simplified curve
    
    fig.add_trace(
        go.Scatter(
            x=vol_range,
            y=frontier_return,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.4)', dash='dash'),
            name='Illustrative Efficient Frontier'
        )
    )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_traces(
        textposition='top center',
        marker=dict(
            line=dict(width=2, color='DarkSlateGrey')
        )
    )
    
    fig.update_layout(
        height=700,
        xaxis_title='Annualized Volatility',
        yaxis_title='Annualized Return'
    )
    
    return fig

def calculate_cluster_stats(metrics_df):
    """
    Calculate aggregate statistics for each cluster
    
    Parameters:
    -----------
    metrics_df : pandas.DataFrame
        DataFrame with risk-return metrics and cluster information
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with average metrics for each cluster
    """
    # Group by cluster and calculate mean statistics
    cluster_stats = metrics_df.groupby('Cluster').agg({
        'Annualized Return': 'mean',
        'Annualized Volatility': 'mean',
        'Sharpe Ratio': 'mean',
        'Fund': 'count'
    }).reset_index()
    
    # Rename the count column
    cluster_stats = cluster_stats.rename(columns={'Fund': 'Number of Funds'})
    
    return cluster_stats

def calculate_cluster_returns(returns_df, cluster_dict):
    """
    Calculate average returns for each cluster
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    cluster_dict : dict
        Dictionary mapping fund IDs to cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with average returns for each cluster
    """
    # Create a dictionary to store returns by cluster
    cluster_returns = {}
    
    # Group funds by cluster
    clusters = {}
    for fund, cluster in cluster_dict.items():
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(fund)
    
    # Calculate average return for each cluster
    for cluster, funds in clusters.items():
        # Filter for funds that exist in returns_df
        valid_funds = [f for f in funds if f in returns_df.columns]
        
        if valid_funds:
            # Calculate average return for each date
            cluster_return = returns_df[valid_funds].mean(axis=1)
            cluster_returns[f"Cluster {cluster}"] = cluster_return
    
    # Create DataFrame from the dictionary
    cluster_returns_df = pd.DataFrame(cluster_returns)
    
    return cluster_returns_df

def create_cumulative_returns_plot(cluster_returns_df):
    """
    Create a cumulative returns plot for clusters
    
    Parameters:
    -----------
    cluster_returns_df : pandas.DataFrame
        DataFrame with returns for each cluster
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Calculate cumulative returns
    cum_returns = np.exp(cluster_returns_df.cumsum()) - 1  # Convert to percentage returns
    
    # Create plot
    fig = px.line(
        cum_returns,
        title="Cumulative Returns by Cluster"
    )
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )
    
    return fig