import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Import functionality from our custom modules
from data_retrieval import setup_eikon, get_fundnav, process_uploaded_isins, normalize_nav_data
from clustering import perform_hierarchical_clustering, perform_kmeans_clustering, perform_knn_clustering, calculate_optimal_clusters
from risk_return import calculate_cluster_metrics, create_risk_return_scatter, calculate_cluster_stats, calculate_cluster_returns, create_cumulative_returns_plot
from utils import calculate_log_returns, calculate_annualized_metrics, display_metrics_table

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
if 'cluster_metrics_df' not in st.session_state:
    st.session_state.cluster_metrics_df = None

# Sidebar for API configuration and inputs
with st.sidebar:
    st.subheader("API Configuration")
    eikon_api_key = st.text_input("Eikon API Key", type="password")
    
    if eikon_api_key:
        try:
            setup_eikon(eikon_api_key)
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
            isins = process_uploaded_isins(uploaded_file)
            if isins:
                st.success(f"Loaded {len(isins)} ISINs from file")
        else:
            isins = []
    
    st.subheader("Time Period")
    weeks_lookback = st.slider("Weeks of historical data", min_value=12, max_value=260, value=52)

    st.subheader("Clustering Parameters")
    clustering_method = st.selectbox(
        "Clustering Method", 
        ["Hierarchical", "K-Means", "K-Nearest Neighbors"]
    )

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
                        normalized_nav = normalize_nav_data(nav_data)
                        
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
        
        # Option to calculate optimal number of clusters
        use_optimal_clusters = st.checkbox("Find optimal number of clusters", value=False)
        
        if use_optimal_clusters:
            with st.spinner("Calculating optimal number of clusters..."):
                # Map string method name to function parameter
                method_map = {
                    "Hierarchical": "hierarchical",
                    "K-Means": "kmeans",
                    "K-Nearest Neighbors": "knn"
                }
                
                # Calculate optimal clusters
                optimal_n, silhouette_scores = calculate_optimal_clusters(
                    returns_df, 
                    max_clusters=min(10, len(returns_df.columns)-1),
                    method=method_map.get(clustering_method, "kmeans")
                )
                
                st.success(f"Optimal number of clusters: {optimal_n}")
                
                # Store silhouette scores in session state
                st.session_state.silhouette_scores = silhouette_scores
                
                # Default to optimal number
                n_clusters = optimal_n
        else:
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
                    if not any(n == n_clusters for n, _ in st.session_state.silhouette_scores):
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
                    if not any(n == n_clusters for n, _ in st.session_state.silhouette_scores):
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
                
                # Calculate metrics for each fund and add cluster info
                cluster_metrics_df = calculate_cluster_metrics(
                    returns_df, 
                    clustering_results['cluster_dict']
                )
                
                # Add fund names to the metrics dataframe
                cluster_metrics_df['Fund Name'] = cluster_metrics_df['Fund'].map(
                    lambda x: st.session_state.fund_names.get(x, x)
                )
                
                # Store in session state
                st.session_state.cluster_metrics_df = cluster_metrics_df
                
                # Display cluster summary
                st.subheader("Cluster Summary")
                
                # Display cluster stats
                for cluster in sorted(cluster_metrics_df['Cluster'].unique()):
                    with st.expander(f"Cluster {cluster}"):
                        cluster_subset = cluster_metrics_df[cluster_metrics_df['Cluster'] == cluster]
                        st.dataframe(cluster_subset.sort_values('Sharpe Ratio', ascending=False))

# Risk-Return Analysis Tab
with tab3:
    st.header("Risk-Return Analysis")
    
    if st.session_state.fund_returns is None:
        st.info("Please fetch fund data in the Data Retrieval tab first.")
    elif st.session_state.clustering_results is None:
        st.info("Please perform clustering analysis in the Clustering Analysis tab first.")
    else:
        # Use cluster information from clustering tab
        returns_df = st.session_state.fund_returns
        cluster_dict = st.session_state.clustering_results['cluster_dict']
        
        # If cluster metrics weren't calculated, do it now
        if st.session_state.cluster_metrics_df is None:
            cluster_metrics_df = calculate_cluster_metrics(returns_df, cluster_dict)
            cluster_metrics_df['Fund Name'] = cluster_metrics_df['Fund'].map(
                lambda x: st.session_state.fund_names.get(x, x)
            )
            st.session_state.cluster_metrics_df = cluster_metrics_df
        else:
            cluster_metrics_df = st.session_state.cluster_metrics_df
        
        # Create risk-return scatter plot
        st.subheader("Risk-Return Scatter Plot")
        
        # Create scatter plot
        fig = create_risk_return_scatter(
            cluster_metrics_df, 
            fund_names=st.session_state.fund_names
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate cluster stats
        st.subheader("Cluster Risk-Return Statistics")
        
        # Group by cluster and calculate mean statistics
        cluster_stats = calculate_cluster_stats(cluster_metrics_df)
        
        # Display cluster statistics
        st.dataframe(cluster_stats.sort_values('Cluster'))
        
        # Cluster performance comparison
        st.subheader("Cluster Performance Comparison")
        
        # Calculate average returns for each cluster
        cluster_returns_df = calculate_cluster_returns(returns_df, cluster_dict)
        
        # Plot cumulative returns
        fig = create_cumulative_returns_plot(cluster_returns_df)
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Nothing to do here - Streamlit runs the script directly
    pass