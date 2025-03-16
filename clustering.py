import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

def perform_hierarchical_clustering(returns_df, n_clusters):
    """
    Perform hierarchical clustering on return data
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    dict
        Dictionary containing clustering results
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

def perform_kmeans_clustering(returns_df, n_clusters):
    """
    Perform K-Means clustering on return data
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    dict
        Dictionary containing clustering results
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

def perform_knn_clustering(returns_df, n_clusters):
    """
    Perform K-Nearest Neighbors clustering on return data
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    n_clusters : int
        Number of clusters to form
        
    Returns:
    --------
    dict
        Dictionary containing clustering results
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

def calculate_optimal_clusters(returns_df, max_clusters=10, method='kmeans'):
    """
    Calculate optimal number of clusters using silhouette scores
    
    Parameters:
    -----------
    returns_df : pandas.DataFrame
        DataFrame containing log returns with dates as index and ISINs as columns
    max_clusters : int, optional
        Maximum number of clusters to evaluate, by default 10
    method : str, optional
        Clustering method ('kmeans', 'knn', or 'hierarchical'), by default 'kmeans'
        
    Returns:
    --------
    tuple
        (optimal_n_clusters, silhouette_scores) where silhouette_scores is a list of 
        (n_clusters, score) tuples for each evaluated cluster count
    """
    max_clusters = min(max_clusters, len(returns_df.columns) - 1)
    silhouette_scores = []
    
    # Calculate silhouette score for each number of clusters
    for n in range(2, max_clusters + 1):
        if method == 'kmeans':
            clustering_results = perform_kmeans_clustering(returns_df, n)
        elif method == 'knn':
            clustering_results = perform_knn_clustering(returns_df, n)
        elif method == 'hierarchical':
            # For hierarchical, we need to do extra work to get silhouette score
            clustering_results = perform_hierarchical_clustering(returns_df, n)
            
            # Transpose returns to have funds as rows for silhouette calculation
            returns_transpose = returns_df.T
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_transpose)
            
            # Get cluster labels
            labels = list(clustering_results['cluster_dict'].values())
            
            # Calculate silhouette score
            clustering_results['silhouette'] = silhouette_score(
                returns_scaled, labels
            ) if n > 1 and n < len(returns_df.columns) else 0
        
        silhouette_scores.append((n, clustering_results['silhouette']))
    
    # Find optimal number of clusters (highest silhouette score)
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0] if silhouette_scores else 2
    
    return optimal_n_clusters, silhouette_scores