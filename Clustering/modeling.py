from sklearn.cluster import KMeans
from utils.base import _Base  # Metaclass providing logger injection

class ClusteringModel(metaclass=_Base):
    """
    KMeans clustering of defensive features.
    """
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        """
        Args:
            n_clusters (int): Number of clusters.
            random_state (int): Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit_predict(self, df):
        """
        Fit KMeans on normalized features and assign cluster labels.

        Returns:
            pd.DataFrame: Input DataFrame with a new 'cluster' column.
        """
        self.logger.info("Starting clustering")
        try:
            feature_cols = [c for c in df.columns if c.startswith('norm_') and c != 'defense_score']
            model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
            df['cluster'] = model.fit_predict(df[feature_cols])
            self.logger.info("Clustering completed")
            return df
        except Exception as e:
            self.logger.error(f"Error during clustering: {e}")
            raise
