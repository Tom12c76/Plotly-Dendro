import pandas as pd
import numpy as np
import streamlit as st

def format_percentage(value, decimals=2):
    """Format a value as a percentage string with specified decimal places"""
    return f"{value * 100:.{decimals}f}%"

def calculate_log_returns(nav_df):
    """
    Calculate logarithmic daily returns from NAV dataframe
    
    Parameters:
    -----------
    nav_df : pandas.DataFrame
        DataFrame containing NAV values with dates as index and ISINs as columns
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame of log returns
    """
    if nav_df is None or nav_df.empty:
        return None
        
    # Calculate log returns
    log_returns = np.log(nav_df / nav_df.shift(1))
    
    # Drop first row with NaN values
    log_returns = log_returns.dropna()
    
    return log_returns

def calculate_annualized_metrics(returns_df):
    """
    Calculate annualized return, volatility, and Sharpe ratio
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns
    
    Returns:
    --------
    dict
        Dictionary containing annualized metrics for each column in returns_df
    """
    metrics = {}
    
    for column in returns_df.columns:
        returns = returns_df[column].dropna()
        
        if len(returns) > 1:
            # Calculate annualized return and volatility (assuming 252 trading days)
            ann_return = returns.mean() * 252
            ann_volatility = returns.std() * np.sqrt(252)
            sharpe = ann_return / ann_volatility if ann_volatility > 0 else 0
            
            metrics[column] = {
                'annualized_return': ann_return,
                'annualized_volatility': ann_volatility,
                'sharpe_ratio': sharpe
            }
    
    return metrics

def display_metrics_table(metrics_dict, fund_names=None):
    """
    Display a formatted metrics table in Streamlit
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metrics from calculate_annualized_metrics
    fund_names : dict, optional
        Dictionary mapping fund IDs to display names
    """
    data = []
    
    for fund_id, metrics in metrics_dict.items():
        display_name = fund_names.get(fund_id, fund_id) if fund_names else fund_id
        
        data.append({
            'Fund': display_name,
            'Annualized Return': format_percentage(metrics['annualized_return']),
            'Annualized Volatility': format_percentage(metrics['annualized_volatility']),
            'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}"
        })
    
    metrics_df = pd.DataFrame(data)
    st.dataframe(metrics_df.sort_values('Sharpe Ratio', ascending=False))