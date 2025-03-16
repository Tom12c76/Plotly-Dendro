import streamlit as st
import pandas as pd
import numpy as np
import eikon as ek
from datetime import datetime, timedelta
import io
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Portfolio Comparison Analysis",
    page_icon="📊",
    layout="wide"
)

# Set title and description
st.title("Portfolio Comparison Analysis")
st.markdown("Compare client portfolio performance against model portfolios")

# Initialize session state for storing data
if 'historical_prices' not in st.session_state:
    st.session_state.historical_prices = {}
if 'client_portfolio' not in st.session_state:
    st.session_state.client_portfolio = None
if 'model_portfolios' not in st.session_state:
    st.session_state.model_portfolios = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'regression_results_client' not in st.session_state:
    st.session_state.regression_results_client = None
if 'regression_results_benchmark' not in st.session_state:
    st.session_state.regression_results_benchmark = None

# Sidebar for API configuration
with st.sidebar:
    st.subheader("API Configuration")
    eikon_api_key = st.text_input("Eikon API Key", type="password")
    
    if eikon_api_key:
        try:
            ek.set_app_key(eikon_api_key)
            st.success("Eikon API Key set successfully!")
        except Exception as e:
            st.error(f"Error setting Eikon API key: {e}")

# Function to fetch historical prices from Eikon
@st.cache_data(ttl=3600)
def fetch_prices(isin_list, periods=110, interval='daily'):
    """
    Fetch historical prices for a list of ISINs from Refinitiv Eikon using get_data
    """
    try:
        # Calculate the start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=periods)
        
        # Format dates for Eikon API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Define fields to retrieve: NAV and Date
        fields = ['tr.fundnav', 'tr.fundnav.date']
        
        # Retrieve data for all ISINs using get_data
        df_hist, err = ek.get_data(isin_list, fields, {
            'SDate': start_date_str, 
            'EDate': end_date_str, 
            'Frq': 'D'  # Daily frequency
        })
        
        # Check for errors
        if err:
            st.warning(f"API warning: {err}")
        
        if df_hist is None or df_hist.empty:
            st.error("No data returned from Eikon API")
            return {}
        
        # Rename columns for clarity
        df_hist = df_hist.rename(columns={'Instrument': 'ISIN', 'NAV': 'CLOSE', 'Date': 'DATE'})
        
        # Convert the 'DATE' column to datetime format
        df_hist['DATE'] = pd.to_datetime(df_hist['DATE'])
        
        # Pivot the dataframe to get ISINs as columns and dates as index
        df_wide = df_hist.pivot(index='DATE', columns='ISIN', values='CLOSE')
        
        # Create a dictionary of price dataframes, one for each ISIN
        all_prices = {}
        for isin in df_wide.columns:
            # Create a dataframe with CLOSE price for this ISIN
            isin_prices = pd.DataFrame(df_wide[isin]).rename(columns={isin: 'CLOSE'})
            # Filter out any NaN values
            isin_prices = isin_prices.dropna()
            # Only add non-empty dataframes
            if not isin_prices.empty:
                all_prices[isin] = isin_prices
        
        return all_prices
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return {}

# Function to parse client portfolio input
def parse_client_portfolio(text_input):
    """
    Parse client portfolio input from text area
    Format expected: ISIN,Units
    """
    lines = text_input.strip().split('\n')
    portfolio = {}
    
    for line in lines:
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                isin = parts[0].strip()
                try:
                    units = float(parts[1].strip())
                    portfolio[isin] = units
                except ValueError:
                    st.warning(f"Invalid units value for {isin}: {parts[1]}")
    
    return portfolio

# Function to calculate portfolio values
def calculate_portfolio_values(prices_dict, portfolio_dict):
    """
    Calculate daily portfolio values based on historical prices and units
    """
    # Initialize an empty dataframe to store combined portfolio values
    portfolio_values = pd.DataFrame()
    
    # For each asset in the portfolio
    for isin, units in portfolio_dict.items():
        if isin in prices_dict:
            # Get asset price history
            asset_prices = prices_dict[isin].copy()
            # Calculate asset value history
            asset_prices['VALUE'] = asset_prices['CLOSE'] * units
            
            # Add to portfolio values
            if portfolio_values.empty:
                portfolio_values = asset_prices[['VALUE']].rename(columns={'VALUE': isin})
            else:
                portfolio_values[isin] = asset_prices['VALUE']
    
    # Calculate total portfolio value
    if not portfolio_values.empty:
        portfolio_values['TOTAL'] = portfolio_values.sum(axis=1)
        
    return portfolio_values

# Function to calculate returns
def calculate_returns(values_df):
    """
    Calculate logarithmic daily returns from value series
    """
    # Make a copy to avoid modifying original
    returns_df = values_df.copy()
    
    # Calculate log returns for each column
    for column in returns_df.columns:
        returns_df[column] = np.log(returns_df[column] / returns_df[column].shift(1))
    
    # Drop first row with NaN values
    returns_df = returns_df.dropna()
    
    return returns_df

# Function to run regression analysis
def run_regression(X, y):
    """
    Run linear regression and return results
    """
    # Make sure we have valid data
    if X.empty or y.empty:
        return None
        
    # Drop any rows with NaN
    valid_data = pd.concat([X, y], axis=1).dropna()
    
    if len(valid_data) < 10:  # Require at least 10 data points
        return None
        
    X = valid_data[X.columns]
    y = valid_data[y.name]
    
    # Run regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared
    r_squared = model.score(X, y)
    
    # Calculate coefficients
    coefficients = pd.Series(model.coef_, index=X.columns)
    
    # Add intercept
    coefficients['intercept'] = model.intercept_
    
    # Calculate predicted values and residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    return {
        'model': model,
        'r_squared': r_squared,
        'coefficients': coefficients,
        'residuals': residuals,
        'y_pred': y_pred,
        'X': X,
        'y': y
    }

# Function to reconstruct model portfolio value
def reconstruct_model_portfolio(model_portfolio, client_total_value):
    """
    Reconstruct model portfolio value by applying weights to model portfolio
    and preserving the actual performance difference compared to client portfolio
    """
    model_values = pd.DataFrame(index=client_total_value.index)
    
    # Get the weights
    isin_weights = model_portfolio['Weight'].to_dict()
    
    # For each asset in the model portfolio
    for isin, weight in isin_weights.items():
        if isin in st.session_state.historical_prices:
            # Get asset price history
            asset_prices = st.session_state.historical_prices[isin].copy()
            # Calculate normalized price * weight
            base_value = asset_prices.iloc[0]['CLOSE']
            asset_prices['NORM_VALUE'] = asset_prices['CLOSE'] / base_value * weight
            
            # Add to model values
            if model_values.empty:
                model_values = asset_prices[['NORM_VALUE']].rename(columns={'NORM_VALUE': isin})
            else:
                model_values[isin] = asset_prices['NORM_VALUE']
    
    # Calculate total model value
    if not model_values.empty:
        model_values['TOTAL'] = model_values.sum(axis=1)
        
        # Scale model to match client portfolio's starting value instead of ending value
        # This preserves the relative performance difference between the portfolios
        client_start_value = client_total_value.iloc[0]
        model_start_value = model_values['TOTAL'].iloc[0]
        
        if model_start_value > 0:  # Avoid division by zero
            scaling_factor = client_start_value / model_start_value
            model_values = model_values * scaling_factor
        else:
            # If we can't properly scale, create a default with client values
            model_values['TOTAL'] = client_total_value.copy()
    else:
        # Create a default TOTAL column with same values as client portfolio
        # to avoid KeyError when no valid assets are found
        model_values['TOTAL'] = client_total_value.copy()
        
    return model_values

# Function to calculate model portfolio returns from log returns of constituents
def calculate_model_returns(historical_prices, model_weights):
    """
    Calculate model portfolio log returns using individual asset log returns and weights
    
    Parameters:
    historical_prices (dict): Dictionary of historical price dataframes for each ISIN
    model_weights (dict): Dictionary with ISIN as key and weight as value
    
    Returns:
    pd.DataFrame: Dataframe with model portfolio log returns and cumulative performance
    """
    # Create an empty dataframe for individual asset returns
    all_returns = pd.DataFrame()
    
    # Calculate log returns for each model constituent
    for isin, weight in model_weights.items():
        if isin in historical_prices:
            # Get prices for this ISIN
            prices = historical_prices[isin].copy()
            
            # Calculate log returns
            returns = np.log(prices['CLOSE'] / prices['CLOSE'].shift(1)).dropna()
            
            # Add to returns dataframe
            if all_returns.empty:
                all_returns = pd.DataFrame(returns)
                all_returns.columns = [isin]
            else:
                all_returns[isin] = returns
    
    if all_returns.empty:
        return pd.DataFrame()
    
    # Create a weights Series
    valid_isins = [isin for isin in model_weights.keys() if isin in all_returns.columns]
    weights = pd.Series([model_weights[isin] for isin in valid_isins], index=valid_isins)
    
    # Normalize weights to sum to 1 (in case some assets are missing)
    weights = weights / weights.sum()
    
    # Select only the columns that exist in both returns and weights
    returns_subset = all_returns[valid_isins]
    
    # Calculate weighted sum of returns (dot product of returns and weights)
    model_returns = pd.DataFrame(index=returns_subset.index)
    model_returns['TOTAL'] = returns_subset.dot(weights)
    
    # Calculate cumulative returns (for visualization)
    model_returns['CUMULATIVE'] = np.exp(model_returns['TOTAL'].cumsum())
    
    return model_returns

# Main app layout
tab1, tab2, tab3 = st.tabs(["Data Input", "Analysis", "Results"])

# Data Input Tab
with tab1:
    st.header("Data Input")
    
    # Section for model portfolios
    st.subheader("Model Portfolios")
    model_file = st.file_uploader("Upload CSV file with model portfolios", type="csv")
    
    if model_file:
        try:
            model_data = pd.read_csv(model_file)
            st.session_state.model_portfolios = model_data
            
            # Display uploaded model portfolios
            # st.dataframe(model_data)
            
            # Get unique model names for selection
            model_names = model_data['Model'].unique().tolist()
            
            # Store the model names in session state
            st.session_state.model_names = model_names
            
        except Exception as e:
            st.error(f"Error processing model portfolios file: {e}")
    else:
        # Provide a sample format
        st.markdown("""
        **Expected CSV format:**
        ```
        Model,ISIN,Weight
        Growth,ABC123,0.25
        Growth,DEF456,0.25
        Growth,GHI789,0.50
        Conservative,JKL012,0.40
        Conservative,MNO345,0.60
        ```
        """)
    
    # Section for client portfolio
    st.subheader("Client Portfolio")
    input_method = st.radio("Input Method", ["Text Input", "Upload File"])
    
    if input_method == "Text Input":
        client_input = st.text_area(
            "Enter client portfolio (ISIN, Units):",
            """ABC123,100
DEF456,200
GHI789,150"""
        )
        
        if st.button("Process Client Input"):
            client_portfolio = parse_client_portfolio(client_input)
            st.session_state.client_portfolio = client_portfolio
            
            # Display parsed client portfolio
            st.dataframe(pd.DataFrame(list(client_portfolio.items()), 
                                     columns=["ISIN", "Units"]))
    else:
        client_file = st.file_uploader("Upload client portfolio file", type=["csv", "txt"])
        
        if client_file:
            client_input = io.StringIO(client_file.getvalue().decode("utf-8")).read()
            client_portfolio = parse_client_portfolio(client_input)
            st.session_state.client_portfolio = client_portfolio
            
            # Display parsed client portfolio
            st.dataframe(pd.DataFrame(list(client_portfolio.items()), 
                                     columns=["ISIN", "Units"]))
    
    # Button to fetch data and proceed to analysis
    if (st.session_state.client_portfolio is not None and 
        isinstance(st.session_state.model_portfolios, pd.DataFrame) and 
        not st.session_state.model_portfolios.empty):
        if st.button("Fetch Historical Data"):
            # Get all unique ISINs
            all_isins = list(st.session_state.client_portfolio.keys())
            all_isins.extend(st.session_state.model_portfolios['ISIN'].unique())
            all_isins = list(set(all_isins))  # Remove duplicates
            
            with st.spinner("Fetching historical prices from Eikon..."):
                # Check if Eikon API key is provided
                if not eikon_api_key:
                    st.error("Please enter your Eikon API key in the sidebar")
                else:
                    # Fetch prices for all ISINs
                    st.session_state.historical_prices = fetch_prices(all_isins)
                    
                    if st.session_state.historical_prices:
                        st.success(f"Successfully fetched data for {len(st.session_state.historical_prices)} assets out of {len(all_isins)} requested")
                        
                        # Create a summary dataframe to show which ISINs have data and which are missing
                        data_status = []
                        for isin in all_isins:
                            is_client = isin in st.session_state.client_portfolio
                            is_model = isin in st.session_state.model_portfolios['ISIN'].values
                            has_data = isin in st.session_state.historical_prices
                            
                            # If we have data, include the date range
                            date_range = None
                            if has_data:
                                prices = st.session_state.historical_prices[isin]
                                start_date = prices.index.min().strftime('%Y-%m-%d')
                                end_date = prices.index.max().strftime('%Y-%m-%d')
                                date_range = f"{start_date} to {end_date}"
                                data_points = len(prices)
                            
                            data_status.append({
                                'ISIN': isin,
                                'In Client Portfolio': '✓' if is_client else '✗',
                                'In Model Portfolio': '✓' if is_model else '✗',
                                'Data Retrieved': '✓' if has_data else '✗',
                                'Date Range': date_range if has_data else 'No data',
                                'Data Points': data_points if has_data else 0
                            })
                        
                        status_df = pd.DataFrame(data_status)
                        
                        # Display the summary table
                        st.subheader("Historical Data Retrieval Summary")
                        st.dataframe(status_df)
                        
                        # Highlight missing data that might cause problems
                        missing_client = [isin for isin in st.session_state.client_portfolio if isin not in st.session_state.historical_prices]
                        missing_model = [isin for isin in st.session_state.model_portfolios['ISIN'].unique() if isin not in st.session_state.historical_prices]
                        
                        if missing_client:
                            st.warning(f"⚠️ Warning: Missing historical data for the following ISINs in client portfolio: {', '.join(missing_client)}")
                        
                        if missing_model:
                            st.warning(f"⚠️ Warning: Missing historical data for the following ISINs in model portfolio: {', '.join(missing_model)}")
                        
                        # Add visualization of data availability
                        if st.session_state.historical_prices:
                            st.subheader("Data Availability Timeline")
                            # Create a figure showing data availability over time for each ISIN
                            fig = go.Figure()
                            
                            for isin, prices in st.session_state.historical_prices.items():
                                # Create binary data availability indicator (1 where data exists)
                                availability = pd.DataFrame(index=prices.index)
                                availability['available'] = 1
                                
                                # Add trace for this ISIN
                                is_client = isin in st.session_state.client_portfolio
                                is_model = isin in st.session_state.model_portfolios['ISIN'].values
                                
                                # Determine color based on where the ISIN is used
                                if is_client and is_model:
                                    color = 'green'  # Both client and model
                                    name = f"{isin} (Client & Model)"
                                elif is_client:
                                    color = 'blue'  # Client only
                                    name = f"{isin} (Client)"
                                else:
                                    color = 'orange'  # Model only
                                    name = f"{isin} (Model)"
                                
                                fig.add_trace(go.Scatter(
                                    x=availability.index, 
                                    y=[isin] * len(availability),
                                    mode='markers',
                                    marker=dict(color=color, size=10),
                                    name=name
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                title='Historical Data Availability by ISIN',
                                xaxis_title='Date',
                                yaxis_title='ISIN',
                                height=max(400, len(st.session_state.historical_prices) * 30),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to fetch historical data. Please check your Eikon API key and ISINs.")

# Analysis Tab
with tab2:
    st.header("Portfolio Analysis")
    
    if not st.session_state.historical_prices:
        st.info("Please fetch historical data in the Data Input tab first")
    else:
        # Select model portfolio for comparison
        if 'model_names' in st.session_state:
            selected_model = st.selectbox("Select Model Portfolio for Comparison", 
                                         st.session_state.model_names)
            
            st.session_state.selected_model = selected_model
            
            # Filter model data for selected model
            model_data = st.session_state.model_portfolios[
                st.session_state.model_portfolios['Model'] == selected_model
            ]
            
            # Display selected model
            st.subheader(f"Selected Model: {selected_model}")
            # st.dataframe(model_data)
            
            # Calculate client portfolio values
            client_values = calculate_portfolio_values(
                st.session_state.historical_prices,
                st.session_state.client_portfolio
            )
            
            # Calculate client returns
            client_returns = calculate_returns(client_values)
            
            # Create weights dictionary for the model portfolio
            model_weights = dict(zip(model_data['ISIN'], model_data['Weight']))
            
            # Calculate model returns using the new function
            model_returns_df = calculate_model_returns(
                st.session_state.historical_prices,
                model_weights
            )
            
            # Store in session state
            st.session_state.client_values = client_values
            st.session_state.client_returns = client_returns
            st.session_state.model_returns = model_returns_df
            
            # Display client portfolio calculation details
            st.subheader("Client Portfolio Calculation")
            
            # Show individual assets and their contribution
            client_assets_fig = go.Figure()
            
            # Add trace for each asset in client portfolio
            for isin in st.session_state.client_portfolio.keys():
                if isin in client_values.columns:
                    client_assets_fig.add_trace(go.Scatter(
                        x=client_values.index,
                        y=client_values[isin],
                        mode='lines',
                        name=f'{isin} ({st.session_state.client_portfolio[isin]} units)',
                        stackgroup='client'  # This creates the stacked area chart
                    ))
            
            # Add total value line
            client_assets_fig.add_trace(go.Scatter(
                x=client_values.index,
                y=client_values['TOTAL'],
                mode='lines',
                name='Total Value',
                line=dict(color='black', width=3)
            ))
            
            # Update layout
            client_assets_fig.update_layout(
                title='Client Portfolio Value Breakdown by Asset',
                xaxis_title='Date',
                yaxis_title='Value',
                legend_title='Assets',
                height=500
            )
            
            st.plotly_chart(client_assets_fig, use_container_width=True)
            
            # Show client portfolio calculation table
            with st.expander("View Client Portfolio Value Calculation Details"):
                # Calculate the latest values for display
                if not client_values.empty:
                    latest_date = client_values.index.max()
                    latest_values = client_values.loc[latest_date].drop('TOTAL')
                    
                    client_calc_data = []
                    for isin in latest_values.index:
                        if isin in st.session_state.client_portfolio and isin in st.session_state.historical_prices:
                            latest_price = st.session_state.historical_prices[isin].loc[latest_date, 'CLOSE']
                            units = st.session_state.client_portfolio[isin]
                            value = latest_values[isin]
                            
                            client_calc_data.append({
                                'ISIN': isin,
                                'Units': units,
                                'Latest Price': latest_price,
                                'Calculation': f"{units} × {latest_price:.2f}",
                                'Value': value
                            })
                    
                    client_calc_df = pd.DataFrame(client_calc_data)
                    client_calc_df['% of Portfolio'] = client_calc_df['Value'] / client_values.loc[latest_date, 'TOTAL'] * 100
                    
                    st.dataframe(client_calc_df)
                    
                    # Show total
                    st.metric("Total Portfolio Value", f"{client_values.loc[latest_date, 'TOTAL']:.2f}")
            
            # Display model portfolio calculation details
            st.subheader("Model Portfolio Calculation")
            
            if not model_returns_df.empty:
                # Create a figure for model portfolio returns
                model_fig = go.Figure()
                
                # Add trace for model portfolio cumulative returns
                model_fig.add_trace(go.Scatter(
                    x=model_returns_df.index,
                    y=model_returns_df['CUMULATIVE'],
                    mode='lines',
                    name='Model Portfolio',
                    line=dict(color='blue', width=3)
                ))
                
                # Update layout
                model_fig.update_layout(
                    title=f'Model Portfolio ({selected_model}) Cumulative Returns',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (Base=1)',
                    height=500
                )
                
                st.plotly_chart(model_fig, use_container_width=True)
                
                # Display explanation of the calculation
                st.subheader("Model Portfolio Weights and Log Returns")
                
                # Create a table showing the model portfolio weights
                weights_df = pd.DataFrame(list(model_weights.items()), columns=['ISIN', 'Weight'])
                weights_df['Normalized Weight'] = weights_df['Weight'] / weights_df['Weight'].sum()
                weights_df['Weight %'] = weights_df['Normalized Weight'] * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Model Portfolio Weights:")
                    st.dataframe(weights_df)
                
                with col2:
                    # Display model portfolio statistics
                    if 'TOTAL' in model_returns_df:
                        # Calculate some statistics on the model returns
                        mean_return = model_returns_df['TOTAL'].mean()
                        std_return = model_returns_df['TOTAL'].std()
                        annualized_return = mean_return * 252  # Assuming 252 trading days
                        annualized_volatility = std_return * np.sqrt(252)
                        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                        
                        st.write("Model Portfolio Return Statistics:")
                        stats_df = pd.DataFrame({
                            'Metric': ['Mean Daily Return', 'Daily Volatility', 
                                       'Annualized Return', 'Annualized Volatility', 
                                       'Sharpe Ratio (0% Risk-Free)'],
                            'Value': [f"{mean_return:.4%}", f"{std_return:.4%}", 
                                      f"{annualized_return:.2%}", f"{annualized_volatility:.2%}", 
                                      f"{sharpe_ratio:.2f}"]
                        })
                        st.dataframe(stats_df)
            
            # Compare client and model returns
            st.subheader("Returns Comparison")
            
            # Create a combined returns dataframe for comparison
            if not client_returns.empty and not model_returns_df.empty:
                st.dataframe(client_values)
                # Make sure we're comparing returns over the same period
                common_dates = client_returns.index.intersection(model_returns_df.index)
                
                if not common_dates.empty:
                    # Get client and model returns for common dates
                    client_common = client_returns.loc[common_dates, 'TOTAL']
                    model_common = model_returns_df.loc[common_dates, 'TOTAL']
                    
                    # Create a cumulative returns dataframe for both
                    cumulative_returns = pd.DataFrame(index=common_dates)
                    cumulative_returns['Client'] = np.exp(client_common.cumsum())
                    cumulative_returns['Model'] = np.exp(model_common.cumsum())
                    cumulative_returns = cumulative_returns - 1  # Convert to percentage returns
                    
                    # Plot cumulative returns comparison
                    cum_fig = go.Figure()
                    
                    # Add client portfolio cumulative returns
                    cum_fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns['Client'] * 100,  # Convert to percentage
                        mode='lines',
                        name='Client Portfolio',
                        line=dict(width=2)
                    ))
                    
                    # Add model portfolio cumulative returns
                    cum_fig.add_trace(go.Scatter(
                        x=cumulative_returns.index,
                        y=cumulative_returns['Model'] * 100,  # Convert to percentage
                        mode='lines',
                        name='Model Portfolio',
                        line=dict(width=2, dash='dash')
                    ))
                    
                    # Add horizontal line at y=0
                    cum_fig.add_shape(
                        type='line',
                        x0=cumulative_returns.index.min(),
                        y0=0,
                        x1=cumulative_returns.index.max(),
                        y1=0,
                        line=dict(color='gray', width=1, dash='dash')
                    )
                    
                    # Update layout
                    cum_fig.update_layout(
                        title='Cumulative Returns Comparison',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Return (%)',
                        legend_title='Portfolio',
                        height=500
                    )
                    
                    st.plotly_chart(cum_fig, use_container_width=True)
                    
                    # Calculate correlation between client and model returns
                    correlation = client_common.corr(model_common)
                    
                    # Display correlation information
                    st.info(f"Correlation between client and model portfolio returns: {correlation:.4f}")
                    
                    # Add scatter plot of returns
                    scatter_fig = px.scatter(
                        x=client_common,
                        y=model_common,
                        labels={
                            'x': 'Client Portfolio Returns',
                            'y': 'Model Portfolio Returns'
                        },
                        title='Daily Returns Comparison',
                        trendline='ols'
                    )
                    
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    # Prepare regression analysis
                    st.subheader("Regression Analysis Setup")
                    
                    # Client Step - Multi-factor regression
                    model_constituents = model_data['ISIN'].tolist()
                    
                    # Create X matrix with model constituent returns
                    X_client_step = pd.DataFrame()
                    for isin in model_constituents:
                        # Get returns for this ISIN from the historical prices
                        if isin in st.session_state.historical_prices:
                            prices = st.session_state.historical_prices[isin].copy()
                            returns = np.log(prices['CLOSE'] / prices['CLOSE'].shift(1)).dropna()
                            if not returns.empty:
                                X_client_step[isin] = returns
                    
                    # Target is client portfolio aggregate returns
                    y_client_step = client_returns['TOTAL']
                    
                    # Make sure we're using the same dates
                    common_dates = X_client_step.index.intersection(y_client_step.index)
                    if not common_dates.empty:
                        X_client_step = X_client_step.loc[common_dates]
                        y_client_step = y_client_step.loc[common_dates]
                        
                        # Run regression only if we have sufficient data
                        if not X_client_step.empty and len(common_dates) >= 10:
                            client_regression = run_regression(X_client_step, y_client_step)
                            st.session_state.regression_results_client = client_regression
                        else:
                            st.warning("Insufficient data for client regression analysis. Need at least 10 common dates.")
                            st.session_state.regression_results_client = None
                    else:
                        st.warning("No common dates found between model constituents and client portfolio for regression.")
                        st.session_state.regression_results_client = None
                    
                    # Benchmark Step - Multi-factor regression
                    # Create X matrix with client fund returns
                    X_benchmark_step = pd.DataFrame()
                    for isin in st.session_state.client_portfolio.keys():
                        # Get returns for this ISIN from the historical prices
                        if isin in st.session_state.historical_prices:
                            prices = st.session_state.historical_prices[isin].copy()
                            returns = np.log(prices['CLOSE'] / prices['CLOSE'].shift(1)).dropna()
                            if not returns.empty:
                                X_benchmark_step[isin] = returns
                    
                    # Target is model portfolio aggregate returns
                    y_benchmark_step = model_returns_df['TOTAL']
                    
                    # Make sure we're using the same dates
                    common_dates = X_benchmark_step.index.intersection(y_benchmark_step.index)
                    if not common_dates.empty:
                        X_benchmark_step = X_benchmark_step.loc[common_dates]
                        y_benchmark_step = y_benchmark_step.loc[common_dates]
                        
                        # Run regression
                        if not X_benchmark_step.empty and len(common_dates) >= 10:
                            benchmark_regression = run_regression(X_benchmark_step, y_benchmark_step)
                            st.session_state.regression_results_benchmark = benchmark_regression
                            
                            # Only show success if both regressions worked
                            if st.session_state.regression_results_client is not None:
                                st.success("Analysis complete! View results in the Results tab.")
                            else:
                                st.warning("Only benchmark regression completed. Client regression failed.")
                        else:
                            st.warning("Insufficient data for benchmark regression analysis.")
                            st.session_state.regression_results_benchmark = None
                    else:
                        st.warning("No common dates found between client constituents and model portfolio for regression.")
                        st.session_state.regression_results_benchmark = None
                else:
                    st.warning("No common dates found between client and model portfolio for comparison.")
            
        else:
            st.warning("No model portfolios available for selection")

# Results Tab
with tab3:
    st.header("Analysis Results")
    
    if (st.session_state.regression_results_client is None or 
        st.session_state.regression_results_benchmark is None):
        st.info("Please complete the analysis in the Analysis tab first")
    else:
        # Display regression results
        st.subheader("Client Portfolio vs Model Constituents")
        st.write("List of underweight model funds to buy")
        
        # Client regression results
        client_r2 = st.session_state.regression_results_client['r_squared']
        client_coefs = st.session_state.regression_results_client['coefficients']
        
        # Create dataframe for coefficients
        client_coefs_df = pd.DataFrame({
            'Factor': client_coefs.index,
            'Loading': client_coefs.values
        })
        
        # Display R-squared
        st.metric("R-squared", f"{client_r2:.4f}")

        col1, col2 = st.columns(2)
        
        # Display coefficients
        with col1:
            # Create bar chart using plotly with sorted values
            # Filter out intercept and sort by loading values
            client_coefs_data = client_coefs_df[client_coefs_df['Factor'] != 'intercept'].copy()
            client_coefs_data = client_coefs_data.sort_values('Loading', ascending=False)

            # Set colors based on positive/negative values
            colors = ['green' if x < 0 else '#4c78a8' for x in client_coefs_data['Loading']]

            fig_client = px.bar(
                client_coefs_data, 
                x='Factor', 
                y='Loading',
                title='Factor Loadings (Sorted)',
                labels={'Loading': 'Coefficient Value'},
                height=400
            )

            # Update trace colors
            fig_client.update_traces(marker_color=colors)

            st.plotly_chart(fig_client, use_container_width=True)

        # Display residual plot
        with col2:
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=st.session_state.regression_results_client['y_pred'],
                y=st.session_state.regression_results_client['residuals'],
                mode='markers',
                marker=dict(size=8),
                name='Residuals'
            ))
            
            # Add horizontal line at y=0
            fig_resid.add_shape(
                type='line',
                x0=min(st.session_state.regression_results_client['y_pred']),
                y0=0,
                x1=max(st.session_state.regression_results_client['y_pred']),
                y1=0,
                line=dict(color='red', dash='dash')
            )
            
            # Update layout
            fig_resid.update_layout(
                title='Residual Plot',
                xaxis_title='Predicted Value',
                yaxis_title='Residual',
                height=400
            )
            
            st.plotly_chart(fig_resid, use_container_width=True)
        
        # Display coefficient table
        st.dataframe(client_coefs_df)
    
        
        # Benchmark regression results
        st.subheader("Model Portfolio vs Client Constituents")
        st.write("List of overweight client funds to sell")

        
        benchmark_r2 = st.session_state.regression_results_benchmark['r_squared']
        benchmark_coefs = st.session_state.regression_results_benchmark['coefficients']
        
        # Create dataframe for coefficients
        benchmark_coefs_df = pd.DataFrame({
            'Factor': benchmark_coefs.index,
            'Loading': benchmark_coefs.values
        })
        
        
        col1, col2 = st.columns(2)

        # Display R-squared
        st.metric("R-squared", f"{benchmark_r2:.4f}")
        
        # Display coefficients
        with col1:
            # Create bar chart using plotly with sorted values
            # Filter out intercept and sort by loading values
            benchmark_coefs_data = benchmark_coefs_df[benchmark_coefs_df['Factor'] != 'intercept'].copy()
            benchmark_coefs_data = benchmark_coefs_data.sort_values('Loading', ascending=False)

            # Set colors based on positive/negative values
            colors = ['red' if x > 0 else '#4c78a8' for x in benchmark_coefs_data['Loading']]

            fig_benchmark = px.bar(
                benchmark_coefs_data, 
                x='Factor', 
                y='Loading',
                title='Factor Loadings (Sorted)',
                labels={'Loading': 'Coefficient Value'},
                height=400
            )

            # Update trace colors
            fig_benchmark.update_traces(marker_color=colors)

            st.plotly_chart(fig_benchmark, use_container_width=True)
        
        
        with col2:        
            # Create residual plot
            fig_resid_benchmark = go.Figure()
            fig_resid_benchmark.add_trace(go.Scatter(
            x=st.session_state.regression_results_benchmark['y_pred'],
            y=st.session_state.regression_results_benchmark['residuals'],
            mode='markers',
            marker=dict(size=8),
            name='Residuals'
            ))
            
            # Add horizontal line at y=0
            fig_resid_benchmark.add_shape(
            type='line',
            x0=min(st.session_state.regression_results_benchmark['y_pred']),
            y0=0,
            x1=max(st.session_state.regression_results_benchmark['y_pred']),
            y1=0,
            line=dict(color='red', dash='dash')
            )
            
            # Update layout
            fig_resid_benchmark.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Value',
            yaxis_title='Residual',
            height=400
            )
            
            st.plotly_chart(fig_resid_benchmark, use_container_width=True)

            
        # Display coefficient table
        st.dataframe(benchmark_coefs_df)
        
        # Display comprehensive results table
        st.subheader("Comprehensive Analysis Results")
        
        # Create combined results dataframe
        results_table = pd.DataFrame({
            'Metric': ['R-squared', 'Intercept'],
            'Client Step': [client_r2, client_coefs['intercept']],
            'Benchmark Step': [benchmark_r2, benchmark_coefs['intercept']]
        })
        
        # Add factor loadings for client step
        for factor in client_coefs.index:
            if factor != 'intercept':
                new_row = {
                    'Metric': f'Client Step: {factor}',
                    'Client Step': client_coefs[factor],
                    'Benchmark Step': None
                }
                results_table = pd.concat([results_table, pd.DataFrame([new_row])], ignore_index=True)
        
        # Add factor loadings for benchmark step
        for factor in benchmark_coefs.index:
            if factor != 'intercept':
                new_row = {
                    'Metric': f'Benchmark Step: {factor}',
                    'Client Step': None,
                    'Benchmark Step': benchmark_coefs[factor]
                }
                results_table = pd.concat([results_table, pd.DataFrame([new_row])], ignore_index=True)
        
        # Display results table
        st.dataframe(results_table)
        
        # Add download button for results
        csv = results_table.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="portfolio_analysis_results.csv",
            mime="text/csv"
        )
        
        # Display interpretation
        st.subheader("Interpretation")
        
        # Client step interpretation
        st.markdown(f"""
        **Client Portfolio Analysis:**
        
        The client portfolio has an R-squared of {client_r2:.4f} with the selected model portfolio constituents,
        which means that {client_r2*100:.2f}% of the client portfolio's return variation can be explained by
        the model portfolio constituents.
        """)
        
        # Benchmark step interpretation
        st.markdown(f"""
        **Model Portfolio Analysis:**
        
        The model portfolio has an R-squared of {benchmark_r2:.4f} with the client portfolio constituents,
        which means that {benchmark_r2*100:.2f}% of the model portfolio's return variation can be explained by
        the client portfolio constituents.
        """)
        
        # Overall conclusion
        better_match = "client portfolio is well-aligned with the model" if client_r2 > 0.7 else "client portfolio may not be well-aligned with the model"
        
        st.markdown(f"""
        **Conclusion:**
        
        Based on the regression analysis, the {better_match} portfolio '{st.session_state.selected_model}'.
        Consider reviewing the factor loadings to understand which components are driving the portfolio's performance
        and where adjustments might be needed to better align with the model.
        """)