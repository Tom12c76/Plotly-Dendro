import streamlit as st
import pandas as pd
import numpy as np
import eikon as ek
from datetime import datetime, timedelta

def setup_eikon(api_key):
    """
    Set up the Eikon API with the provided key
    
    Parameters:
    -----------
    api_key : str
        Eikon API key
    
    Returns:
    --------
    bool
        True if setup was successful, False otherwise
    """
    try:
        ek.set_app_key(api_key)
        return True
    except Exception as e:
        st.error(f"Error setting Eikon API key: {e}")
        return False

@st.cache_data(ttl=3600)
def get_fundnav(isin_list, weeks=52):
    """
    Fetch historical NAVs for a list of ISINs from Eikon API
    
    Parameters:
    -----------
    isin_list : list
        List of ISINs to fetch NAV data for
    weeks : int, optional
        Number of weeks of historical data to fetch, by default 52
    
    Returns:
    --------
    tuple
        (nav_data, fund_names) where:
        - nav_data is a DataFrame with dates as index and ISINs as columns
        - fund_names is a dict mapping ISINs to fund names
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
            return None, {}
        
        # Convert to datetime and keep only the date component
        fundnav['Date'] = pd.to_datetime(fundnav['Date']).dt.date
        fundnav = fundnav.pivot(index='Date', columns='Instrument', values='NAV')
        
        # Get fund names for better display
        fund_names_data, err = ek.get_data(isin_list, ['TR.FundName'])
        if not err and not fund_names_data.empty:
            fund_names = dict(zip(fund_names_data['Instrument'], fund_names_data['Fund Name']))
            return fundnav, fund_names
        
        return fundnav, {}
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return None, {}

def process_uploaded_isins(uploaded_file):
    """
    Process an uploaded file containing ISINs
    
    Parameters:
    -----------
    uploaded_file : streamlit.UploadedFile
        CSV file uploaded by the user
        
    Returns:
    --------
    list
        List of ISINs
    """
    try:
        df_isins = pd.read_csv(uploaded_file)
        if 'ISIN' in df_isins.columns:
            isins = df_isins['ISIN'].tolist()
        else:
            isins = df_isins.iloc[:, 0].tolist()
        return isins
    except Exception as e:
        st.error(f"Error reading ISIN file: {e}")
        return []

def normalize_nav_data(nav_data):
    """
    Normalize NAV data for better visualization (base=1)
    
    Parameters:
    -----------
    nav_data : pandas.DataFrame
        DataFrame containing NAV values with dates as index and ISINs as columns
    
    Returns:
    --------
    pandas.DataFrame
        Normalized NAV DataFrame
    """
    if nav_data is None or nav_data.empty:
        return None
        
    # Normalize to the first available value
    return nav_data / nav_data.iloc[0]