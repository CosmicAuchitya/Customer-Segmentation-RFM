# -*- coding: utf-8 -*-
"""
Customer Segmentation using K-Means Clustering based on RFM Model.

This script performs the following steps:
1. Data Loading & Cleaning
2. RFM Calculation (Feature Engineering)
3. Preprocessing (Log Transformation & Scaling)
4. Modeling (K-Means Clustering with Elbow Method)
5. Visualization & Interpretation
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import datetime as dt

# Set plot style
sns.set(style="whitegrid")

def main():
    # ==========================================
    # 1. Data Loading & Cleaning
    # ==========================================
    print("Loading dataset...")
    # Load the dataset
    # Ensure 'Online Retail.xlsx' is in the current directory or provide the full path
    try:
        df = pd.read_excel('Online Retail.xlsx')
    except FileNotFoundError:
        print("Error: 'Online Retail.xlsx' not found. Please upload the file.")
        return

    print(f"Initial data shape: {df.shape}")

    # Drop rows where 'CustomerID' is missing
    df.dropna(subset=['CustomerID'], inplace=True)
    print(f"Data shape after dropping missing CustomerID: {df.shape}")

    # Remove rows where 'Quantity' is negative (returns) and 'UnitPrice' is <= 0
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    print(f"Data shape after removing returns and invalid prices: {df.shape}")

    # Create a 'TotalAmount' column
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    print("Data cleaning completed.\n")

    # ==========================================
    # 2. RFM Calculation (Feature Engineering)
    # ==========================================
    print("Calculating RFM metrics...")
    
    # Set reference date as the day after the last transaction
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    print(f"Reference Date: {reference_date}")

    # Calculate RFM metrics
    # Recency: Days since last purchase
    # Frequency: Number of unique transactions (InvoiceNo)
    # Monetary: Sum of TotalAmount
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()

    # Rename columns
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    }, inplace=True)

    print(f"RFM data shape: {rfm.shape}")
    print(rfm.head())
    print("\n")

    # ==========================================
    # 3. Preprocessing
    # ==========================================
    print("Preprocessing data...")

    # Check for skewness and apply Log Transformation
    # Adding a small constant to handle zeroes if any (though Recency/Frequency usually > 0)
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))

    # Scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # Create a DataFrame for scaled data
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    print("Preprocessing completed.\n")

    # ==========================================
    # 4. Modeling (K-Means)
    # ==========================================
    print("Determining optimal clusters (Elbow Method)...")
    
    # Elbow Method
    wcss = []
    range_n_clusters = range(1, 11)
    for k in range_n_clusters:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)

    # Plot Elbow Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range_n_clusters, wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()
    print("Elbow curve displayed. Using k=3 for this example (adjust based on plot).")

    # Apply K-Means with k=3 (You can change this based on the Elbow plot)
    k_optimal = 3
    kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)

    # Add Cluster labels to original RFM dataframe
    rfm['Cluster'] = cluster_labels
    print("Clustering completed.\n")

    # ==========================================
    # 5. Visualization & Interpretation
    # ==========================================
    print("Visualizing and interpreting clusters...")

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'green', 'blue', 'purple']
    
    for i in range(k_optimal):
        cluster_data = rfm[rfm['Cluster'] == i]
        ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], 
                   c=colors[i % len(colors)], label=f'Cluster {i}', s=50, alpha=0.6)

    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('3D Scatter Plot of Customer Segments')
    ax.legend()
    plt.show()

    # Calculate mean of RFM for each Cluster
    cluster_summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'CustomerID': 'count'
    }).round(2)
    
    cluster_summary.rename(columns={'CustomerID': 'Count'}, inplace=True)
    
    print("Cluster Summary:")
    print(cluster_summary)
    
    # Interpretation (Simple logic based on means, can be adjusted)
    print("\nInterpretation:")
    for cluster_id, row in cluster_summary.iterrows():
        print(f"Cluster {cluster_id}: {int(row['Count'])} customers - "
              f"R: {row['Recency']}, F: {row['Frequency']}, M: {row['Monetary']}")

if __name__ == "__main__":
    main()
