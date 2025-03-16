import streamlit as st
import pandas as pd
import numpy as np
import eikon as ek
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Fund Clustering Analysis",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Mutual Fund Clustering Analysis")
st.markdown("""
This app analyzes historical log returns of mutual funds and applies clustering 
techniques to identify funds with similar return patterns. You can select different
clustering methods and visualize the results.
""")

# Initialize session state for storing data
if 'historical_navs' not in st.session_state:
    st.session_state.historical_navs = None
if 'fund_returns' not in st.session_state:
    st.session_state.fund_returns = None
if 'fund_names' not in st.session_state:
    st.session_state.fund_names = {}
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'silhouette_scores' not in st.session_state:
    st.session_state.silhouette_scores = []

# Sidebar for API configuration and inputs
with st.sidebar:
    st.subheader("API Configuration")
    eikon_api_key = st.text_input("Eikon API Key", type="password")
    
    if eikon_api_key:
        try:
            ek.set_app_key(eikon_api_key)
            st.success("Eikon API Key set successfully!")
        except Exception as e:
            st.error(f"Error setting Eikon API key: {e}")
    
    st.subheader("Data Parameters")
    data_input_method = st.radio("Input Method", ["Default Funds", "Enter ISINs", "Upload File"])
    
    if data_input_method == "Default Funds":
        isins = [
            'LP68644630', 'LP68422844', '/WLD.PA', 'LP68569647', 'LP60060386', 'LP60060398', 'LP60060399', 
            'LP68335443', 'LP68335464', 'LP68644626', 'LP68644639', 'LP68644656', 'LP68644670', 'LP68759031'
        ]
    elif data_input_method == "Enter ISINs":
        isin_input = st.text_area(
            "Enter ISINs (one per line):",
            """LP68644630
LP68422844
/WLD.PA
LP68569647
LP60060386"""
        )
        isins = [isin.strip() for isin in isin_input.split("\n") if isin.strip()]
    else:
        uploaded_file = st.file_uploader("Upload a CSV file with ISINs", type="csv")
        if uploaded_file:
            try:
                df_isins = pd.read_csv(uploaded_file)
                if 'ISIN' in df_isins.columns:
                    isins = df_isins['ISIN'].tolist()
                else:
                    isins = df_isins.iloc[:, 0].tolist()
                st.success(f"Loaded {len(isins)} ISINs from file")
            except Exception as e:
                st.error(f"Error reading ISIN file: {e}")
                isins = []
        else:
            isins = []
    
    st.subheader("Time Period")
    weeks_lookback = st.slider("Weeks of historical data", min_value=12, max_value=260, value=52)

    st.subheader("Clustering Parameters")
    clustering_method = st.selectbox(
        "Clustering Method", 
        ["Hierarchical", "K-Means", "K-Nearest Neighbors"]
    )

# Function to fetch historical NAV data from Eikon
@st.cache_data(ttl=3600)
def get_fundnav(isin_list, weeks=52):
    """
    Fetch historical NAVs for a list of ISINs from Eikon API
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
            return None
        
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

# Function to calculate log returns
def calculate_log_returns(nav_df):
    """
    Calculate logarithmic daily returns from NAV dataframe
    """
    if nav_df is None or nav_df.empty:
        return None
        
    # Calculate log returns
    log_returns = np.log(nav_df / nav_df.shift(1))
    
    # Drop first row with NaN values
    log_returns = log_returns.dropna()
    
    return log_returns

# Function to perform hierarchical clustering
def perform_hierarchical_clustering(returns_df, n_clusters):
    """
    Perform hierarchical clustering on return data
    """
    # Calculate the correlation matrix
    corr_matrix = returns_df.corr()
    
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Get cluster labels
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Create a dictionary mapping fund to cluster
    cluster_dict = dict(zip(returns_df.columns, labels))
    
    return {
        'cluster_dict': cluster_dict,
        'linkage_matrix': linkage_matrix,
        'distance_matrix': distance_matrix,
        'corr_matrix': corr_matrix
    }

# Function to perform K-Means clustering
def perform_kmeans_clustering(returns_df, n_clusters):
    """
    Perform K-Means clustering on return data
    """
    # Transpose returns to have funds as rows
    returns_transpose = returns_df.T
    
    # Standardize the data
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_transpose)
    
    # Apply PCA to reduce dimensionality
    if returns_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        returns_pca = pca.fit_transform(returns_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
    else:
        returns_pca = returns_scaled
        explained_var = 1.0
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(returns_scaled)
    
    # Calculate silhouette score if we have enough clusters
    silhouette = silhouette_score(returns_scaled, cluster_labels) if n_clusters > 1 and n_clusters < len(returns_df.columns) else 0
    
    # Create a dictionary mapping fund to cluster
    cluster_dict = dict(zip(returns_df.columns, cluster_labels))
    
    return {
        'cluster_dict': cluster_dict,
        'returns_pca': returns_pca,
        'explained_var': explained_var,
        'silhouette': silhouette,
        'fund_order': returns_df.columns
    }

# Function to perform KNN clustering
def perform_knn_clustering(returns_df, n_clusters):
    """
    Perform K-Nearest Neighbors clustering on return data
    """
    # Transpose returns to have funds as rows
    returns_transpose = returns_df.T
    
    # Standardize the data
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_transpose)
    
    # Apply PCA to reduce dimensionality
    if returns_scaled.shape[1] > 2:
        pca = PCA(n_components=2)
        returns_pca = pca.fit_transform(returns_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
    else:
        returns_pca = returns_scaled
        explained_var = 1.0
    
    # Create KNN model
    knn = NearestNeighbors(n_neighbors=min(n_clusters, len(returns_df.columns)-1))
    knn.fit(returns_scaled)
    
    # Get distances and indices of nearest neighbors
    distances, indices = knn.kneighbors(returns_scaled)
    
    # Perform K-Means on the data to get cluster labels (KNN itself doesn't provide clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(returns_scaled)
    
    # Calculate silhouette score if we have enough clusters
    silhouette = silhouette_score(returns_scaled, cluster_labels) if n_clusters > 1 and n_clusters < len(returns_df.columns) else 0
    
    # Create a dictionary mapping fund to cluster
    cluster_dict = dict(zip(returns_df.columns, cluster_labels))
    
    return {
        'cluster_dict': cluster_dict,
        'returns_pca': returns_pca,
        'explained_var': explained_var,
        'silhouette': silhouette,
        'fund_order': returns_df.columns,
        'distances': distances,
        'indices': indices
    }

# Main app layout
tab1, tab2, tab3 = st.tabs(["Data Retrieval", "Clustering Analysis", "Risk-Return Analysis"])

# Data Retrieval Tab
with tab1:
    st.header("Fund Data Retrieval")
    
    if not isins:
        st.info("Please enter ISINs in the sidebar to fetch data.")
    else:
        st.write(f"Selected {len(isins)} funds for analysis:")
        st.write(", ".join(isins))
        
        if st.button("Fetch Historical NAV Data"):
            with st.spinner("Fetching data from Eikon..."):
                # Check if Eikon API key is provided
                if not eikon_api_key:
                    st.error("Please enter your Eikon API key in the sidebar")
                else:
                    # Fetch NAV data
                    nav_data, fund_names = get_fundnav(isins, weeks=weeks_lookback)
                    
                    if nav_data is not None and not nav_data.empty:
                        # Store in session state
                        st.session_state.historical_navs = nav_data
                        st.session_state.fund_names = fund_names
                        
                        # Calculate log returns
                        log_returns = calculate_log_returns(nav_data)
                        st.session_state.fund_returns = log_returns
                        
                        st.success(f"Successfully retrieved data for {nav_data.shape[1]} funds over {nav_data.shape[0]} days.")
                        
                        # Display NAV data
                        st.subheader("Historical NAV Data")
                        st.dataframe(nav_data.tail(10))
                        
                        # Display log returns
                        st.subheader("Log Returns")
                        st.dataframe(log_returns.tail(10))
                        
                        # Plot NAV data
                        st.subheader("Historical NAV Trends")
                        # Normalize NAV data for better comparison
                        normalized_nav = nav_data / nav_data.iloc[0]
                        
                        fig = px.line(normalized_nav, title="Normalized NAV (Base=1)")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot log returns
                        st.subheader("Log Returns Distribution")
                        fig = go.Figure()
                        
                        for col in log_returns.columns:
                            fig.add_trace(go.Box(
                                y=log_returns[col].dropna(),
                                name=fund_names.get(col, col),
                                boxpoints='outliers'
                            ))
                        
                        fig.update_layout(
                            title="Distribution of Log Returns",
                            yaxis_title="Log Return",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Failed to fetch NAV data. Please check your ISINs and API key.")

# Clustering Analysis Tab
with tab2:
    st.header("Fund Clustering Analysis")
    
    if st.session_state.fund_returns is None:
        st.info("Please fetch fund data in the Data Retrieval tab first.")
    else:
        # Get fund return data
        returns_df = st.session_state.fund_returns
        
        # Slider for number of clusters
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=1,
            max_value=len(returns_df.columns),
            value=min(4, len(returns_df.columns))
        )
        
        # Perform clustering based on selected method
        if st.button("Run Clustering Analysis"):
            with st.spinner(f"Performing {clustering_method} clustering..."):
                if clustering_method == "Hierarchical":
                    clustering_results = perform_hierarchical_clustering(returns_df, n_clusters)
                    st.session_state.clustering_results = clustering_results
                    
                    # Display correlation matrix as heatmap
                    st.subheader("Correlation Matrix")
                    
                    # Create correlation matrix heatmap
                    corr_matrix = clustering_results['corr_matrix']
                    
                    # Replace ISIN with fund name in correlation matrix if available
                    if st.session_state.fund_names:
                        renamed_corr_matrix = corr_matrix.rename(
                            index=lambda x: st.session_state.fund_names.get(x, x),
                            columns=lambda x: st.session_state.fund_names.get(x, x)
                        )
                    else:
                        renamed_corr_matrix = corr_matrix
                    
                    fig = px.imshow(
                        renamed_corr_matrix,
                        labels=dict(x="Fund", y="Fund", color="Correlation"),
                        color_continuous_scale="RdBu_r",
                        title="Fund Returns Correlation Matrix"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create dendrogram
                    st.subheader("Hierarchical Clustering Dendrogram")
                    
                    # Create labels with fund names if available
                    labels = [st.session_state.fund_names.get(col, col) for col in returns_df.columns]
                    
                    fig_dendro = ff.create_dendrogram(
                        clustering_results['distance_matrix'],
                        labels=labels,
                        linkagefun=lambda x: clustering_results['linkage_matrix'],
                        color_threshold=0.7
                    )
                    
                    fig_dendro.update_layout(
                        width=900,
                        height=600,
                        title="Hierarchical Clustering Dendrogram"
                    )
                    
                    st.plotly_chart(fig_dendro, use_container_width=True)
                    
                elif clustering_method == "K-Means":
                    clustering_results = perform_kmeans_clustering(returns_df, n_clusters)
                    st.session_state.clustering_results = clustering_results
                    
                    # Store silhouette score
                    silhouette = clustering_results['silhouette']
                    if len(st.session_state.silhouette_scores) < len(returns_df.columns):
                        st.session_state.silhouette_scores.append((n_clusters, silhouette))
                    
                    # Display 2D PCA plot with clusters
                    st.subheader("K-Means Clustering - PCA Visualization")
                    st.write(f"Explained variance with PCA: {clustering_results['explained_var']:.2%}")
                    st.write(f"Silhouette Score: {silhouette:.4f}")
                    
                    # Create dataframe for PCA visualization
                    pca_df = pd.DataFrame(
                        clustering_results['returns_pca'],
                        columns=['PC1', 'PC2']
                    )
                    pca_df['Fund'] = clustering_results['fund_order']
                    pca_df['Cluster'] = [clustering_results['cluster_dict'][fund] for fund in clustering_results['fund_order']]
                    pca_df['Fund Name'] = [st.session_state.fund_names.get(fund, fund) for fund in clustering_results['fund_order']]
                    
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        hover_name='Fund Name',
                        text='Fund Name',
                        title="K-Means Clustering - PCA Visualization"
                    )
                    
                    fig.update_traces(textposition='top center', marker=dict(size=10))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display silhouette scores for different cluster counts
                    if len(st.session_state.silhouette_scores) > 1:
                        st.subheader("Silhouette Scores for Different Cluster Counts")
                        silhouette_df = pd.DataFrame(
                            st.session_state.silhouette_scores,
                            columns=['Clusters', 'Silhouette Score']
                        )
                        
                        fig = px.line(
                            silhouette_df,
                            x='Clusters',
                            y='Silhouette Score',
                            title="Silhouette Scores by Cluster Count",
                            markers=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                elif clustering_method == "K-Nearest Neighbors":
                    clustering_results = perform_knn_clustering(returns_df, n_clusters)
                    st.session_state.clustering_results = clustering_results
                    
                    # Store silhouette score
                    silhouette = clustering_results['silhouette']
                    if len(st.session_state.silhouette_scores) < len(returns_df.columns):
                        st.session_state.silhouette_scores.append((n_clusters, silhouette))
                    
                    # Display 2D PCA plot with clusters
                    st.subheader("KNN-Based Clustering - PCA Visualization")
                    st.write(f"Explained variance with PCA: {clustering_results['explained_var']:.2%}")
                    st.write(f"Silhouette Score: {silhouette:.4f}")
                    
                    # Create dataframe for PCA visualization
                    pca_df = pd.DataFrame(
                        clustering_results['returns_pca'],
                        columns=['PC1', 'PC2']
                    )
                    pca_df['Fund'] = clustering_results['fund_order']
                    pca_df['Cluster'] = [clustering_results['cluster_dict'][fund] for fund in clustering_results['fund_order']]
                    pca_df['Fund Name'] = [st.session_state.fund_names.get(fund, fund) for fund in clustering_results['fund_order']]
                    
                    # Create a plot with clusters and connections
                    fig = px.scatter(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        hover_name='Fund Name',
                        text='Fund Name',
                        title="KNN-Based Clustering - PCA Visualization"
                    )
                    
                    # Add KNN connections (lines between nearest neighbors)
                    indices = clustering_results['indices']
                    for i, neighbors in enumerate(indices):
                        for neighbor_idx in neighbors[1:3]:  # Only take the 2 closest neighbors
                            fig.add_trace(
                                go.Scatter(
                                    x=[pca_df.iloc[i]['PC1'], pca_df.iloc[neighbor_idx]['PC1']],
                                    y=[pca_df.iloc[i]['PC2'], pca_df.iloc[neighbor_idx]['PC2']],
                                    mode='lines',
                                    line=dict(color='rgba(0,0,0,0.2)', width=1),
                                    showlegend=False
                                )
                            )
                    
                    fig.update_traces(textposition='top center', marker=dict(size=10))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster summary
                st.subheader("Cluster Summary")
                cluster_results = []
                
                for fund, cluster in clustering_results['cluster_dict'].items():
                    fund_name = st.session_state.fund_names.get(fund, fund)
                    fund_returns = returns_df[fund]
                    
                    # Calculate annualized return and volatility
                    ann_return = fund_returns.mean() * 252  # Assuming 252 trading days
                    ann_volatility = fund_returns.std() * np.sqrt(252)
                    sharpe = ann_return / ann_volatility if ann_volatility > 0 else 0
                    
                    cluster_results.append({
                        'Fund': fund,
                        'Fund Name': fund_name,
                        'Cluster': cluster,
                        'Annualized Return': ann_return,
                        'Annualized Volatility': ann_volatility,
                        'Sharpe Ratio': sharpe
                    })
                
                cluster_df = pd.DataFrame(cluster_results)
                
                # Display cluster stats
                for cluster in sorted(cluster_df['Cluster'].unique()):
                    with st.expander(f"Cluster {cluster}"):
                        cluster_subset = cluster_df[cluster_df['Cluster'] == cluster]
                        st.dataframe(cluster_subset.sort_values('Sharpe Ratio', ascending=False))
                
                # Store the cluster results in session state for use in other tabs
                st.session_state.cluster_df = cluster_df

# Risk-Return Analysis Tab
with tab3:
    st.header("Risk-Return Analysis")
    
    if st.session_state.fund_returns is None:
        st.info("Please fetch fund data in the Data Retrieval tab first.")
    elif 'cluster_df' not in st.session_state:
        st.info("Please perform clustering analysis in the Clustering Analysis tab first.")
    else:
        # Use cluster information from previous tab
        cluster_df = st.session_state.cluster_df
        
        # Create risk-return scatter plot
        st.subheader("Risk-Return Scatter Plot")
        
        # Create a scatter plot with clusters
        # FIX: Ensure Sharpe Ratio values are positive for the size parameter
        sharpe_values = cluster_df['Sharpe Ratio'].values
        # Convert to absolute values and add a small constant to ensure all values are positive
        size_values = np.abs(sharpe_values) + 1
        
        fig = px.scatter(
            cluster_df,
            x='Annualized Volatility',
            y='Annualized Return',
            color='Cluster',
            hover_name='Fund Name',
            text='Fund Name',
            # Use the fixed size values
            size=size_values,
            size_max=20,
            title="Risk-Return Analysis by Cluster"
        )
        
        # Add efficient frontier line (simple illustration)
        min_vol = cluster_df['Annualized Volatility'].min() * 0.9
        max_vol = cluster_df['Annualized Volatility'].max() * 1.1
        
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
            height=800,
            xaxis_title='Annualized Volatility',
            yaxis_title='Annualized Return'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate cluster stats
        st.subheader("Cluster Risk-Return Statistics")
        
        # Group by cluster and calculate mean statistics
        cluster_stats = cluster_df.groupby('Cluster').agg({
            'Annualized Return': 'mean',
            'Annualized Volatility': 'mean',
            'Sharpe Ratio': 'mean',
            'Fund': 'count'
        }).reset_index()
        
        cluster_stats = cluster_stats.rename(columns={'Fund': 'Number of Funds'})
        
        # Display cluster statistics
        st.dataframe(cluster_stats.sort_values('Cluster'))
        
        # Cluster performance comparison
        st.subheader("Cluster Performance Comparison")
        
        # Calculate cumulative returns for each cluster
        returns_df = st.session_state.fund_returns
        
        # Create average return series for each cluster
        cluster_returns = {}
        for cluster in sorted(cluster_df['Cluster'].unique()):
            # Get funds in this cluster
            funds_in_cluster = cluster_df[cluster_df['Cluster'] == cluster]['Fund'].tolist()
            
            if funds_in_cluster:
                # Calculate average return for each date
                cluster_return = returns_df[funds_in_cluster].mean(axis=1)
                cluster_returns[f"Cluster {cluster}"] = cluster_return
        
        # Create cumulative return series
        cum_returns = pd.DataFrame(cluster_returns)
        cum_returns = np.exp(cum_returns.cumsum()) - 1  # Convert to percentage returns
        
        # Plot cumulative returns
        fig = px.line(
            cum_returns,
            title="Cumulative Returns by Cluster"
        )
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        
        fig.update_layout(
            height=600,
            xaxis_title="Date",
            yaxis_title="Cumulative Return"
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Nothing to do here - Streamlit runs the script directly
    pass