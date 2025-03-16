import numpy as np
import pandas as pd
import streamlit as st
import eikon as ek
from datetime import datetime, timedelta

ek.set_app_key('cf2eaf5e3b3c42adba08b3c5c2002b6ced1e77d7')
st.success("Eikon API Key set successfully!", icon="✅")

# Function to fetch historical prices from Eikon
# @st.cache_data
def get_fundnav(isin_list, weeks=110, interval='daily'):
    """
    Fetch historical NAVs for a list of ISINs from Eikon using get_data
    """

    try:
        # Calculate the start and end dates
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)
        
        # Format dates for Eikon API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Define fields to retrieve: NAV and Date
        fields = ['TR.FundNAV', 'TR.FundNAV.Date']
        
        # Retrieve data for all ISINs using get_data, specifying EUR as currency
        fundnav, err = ek.get_data(isin_list, fields, {
            'SDate': start_date_str, 
            'EDate': end_date_str, 
            'Frq': 'D',  # Daily frequency
            'Curn': 'EUR'  # Force values in EUR
        })
        
        # Check for errors
        if err:
            st.warning(f"API warning: {err}")
        
        if fundnav is None or fundnav.empty:
            st.error("No data returned from Eikon API")
            return {}
        
        # Convert to datetime and keep only the date component
        fundnav['Date'] = pd.to_datetime(fundnav['Date']).dt.date
        fundnav = fundnav.pivot(index='Date', columns='Instrument', values='NAV')
        
        return fundnav
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return {}

def calculate_portfolio_analysis(client_ptf, fundnav):
    """
    Calculate portfolio analysis metrics including quantity matrix, cash flows, portfolio value, and returns.
    
    Parameters:
    -----------
    client_ptf : pandas.DataFrame
        DataFrame containing the portfolio information with ISIN and Quantity columns
    fundnav : pandas.DataFrame
        DataFrame containing the fund NAV values with dates as index and ISINs as columns
    
    Returns:
    --------
    dict
        A dictionary containing the calculated matrices and metrics:
        - quantity_matrix: Matrix of quantities for each ISIN over time
        - quantity_change: Daily change in quantities
        - cash_flow_matrix: Matrix of cash flows for each ISIN
        - daily_cash_flow: Aggregated daily cash flows
        - portfolio_value: Portfolio value over time
        - logret_ptf: Daily returns adjusted for cash flows
    """
    # Extract quantities from client_ptf
    quantities = client_ptf.set_index('ISIN')['Quantity']
    
    # Create quantity matrix with the same structure as fundnav
    quantity_matrix = pd.DataFrame(index=fundnav.index, columns=fundnav.columns)
    
    # Fill the DataFrame with quantities (vectorized approach)
    for isin in quantity_matrix.columns:
        quantity_matrix[isin] = quantities.get(isin, 0)
    
    # Mask quantity matrix where NAV values exist
    masked_quantity_matrix = quantity_matrix.where(fundnav > 0)
    
    # Calculate quantity changes and cash flows
    quantity_change = masked_quantity_matrix.fillna(0).diff().fillna(0)
    cash_flow_matrix = quantity_change * fundnav
    daily_cash_flow = cash_flow_matrix.sum(axis=1)
    
    # Calculate portfolio value and returns
    portfolio_value = masked_quantity_matrix * fundnav
    portfolio_value['PTF'] = portfolio_value.sum(axis=1)
    
    logret_ptf = np.log((portfolio_value.PTF - daily_cash_flow) / portfolio_value.PTF.shift(1))
    
    return {
        'quantity_matrix': quantity_matrix,
        'masked_quantity_matrix': masked_quantity_matrix,
        'quantity_change': quantity_change,
        'cash_flow_matrix': cash_flow_matrix,
        'daily_cash_flow': daily_cash_flow,
        'portfolio_value': portfolio_value,
        'logret_ptf': logret_ptf
    }
    
client_ptf = pd.read_csv('ISIN_Quantities.csv')

st.dataframe(client_ptf)
fundnav = get_fundnav(client_ptf['ISIN'].tolist())

st.write(fundnav)

# Process the portfolio data
portfolio_data = calculate_portfolio_analysis(client_ptf, fundnav)

# # Display the calculated matrices and metrics
# st.write(portfolio_data['quantity_matrix'])
# st.write(portfolio_data['masked_quantity_matrix'])
# st.write(portfolio_data['quantity_change'])
# st.write(portfolio_data['cash_flow_matrix'])
# st.write(portfolio_data['daily_cash_flow'])
# st.write(portfolio_data['portfolio_value']['PTF'])

# Plot the results
st.line_chart(np.exp(portfolio_data['logret_ptf'].cumsum())-1)
st.line_chart(portfolio_data['daily_cash_flow'].cumsum())
