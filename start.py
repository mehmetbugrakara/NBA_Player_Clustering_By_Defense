from Clustering.reading import DataFetcher
from Clustering.preprocessing import DataPreprocessor
from Clustering.feature_engineering import FeatureEngineer
from Clustering.modeling import ClusteringModel
from utils.base import _Base  # Metaclass providing logger injection

class DefensiveShotClusterPipeline(metaclass=_Base):
    """
    Orchestrator combining all pipeline components.
    """
    def __init__(self, seasons: list, output_path: str,
                 gp_thresh: int = 20, min_thresh: int = 12,
                 recent_seasons: int = 2, n_clusters: int = 10):
        """
        Args:
            seasons (list): Seasons to fetch.
            output_path (str): Path to save results (.feather).
            gp_thresh (int): Games played threshold.
            min_thresh (int): Minutes played threshold.
            recent_seasons (int): Seasons for averaging.
            n_clusters (int): Number of clusters.
        """
        self.fetcher = DataFetcher(seasons)
        self.preprocessor = DataPreprocessor(gp_thresh, min_thresh)
        self.engineer = FeatureEngineer(recent_seasons)
        self.clusterer = ClusteringModel(n_clusters)
        self.output_path = output_path

    def run(self):
        """
        Execute full pipeline: fetch, preprocess, engineer, cluster, save.
        """
        self.logger.info("Pipeline started")
        gen_df, ls6_df, fp_df, tp_df, players_df = self.fetcher.fetch_all()
        pre_df = self.preprocessor.preprocess(gen_df, ls6_df, fp_df, tp_df)
        feat_df = self.engineer.engineer(pre_df)
        result_df = self.clusterer.fit_predict(feat_df)
        result_df.to_excel(self.output_path,index=False)
        self.logger.info(f"Pipeline completed. Results saved to {self.output_path}")


if __name__ == "__main__":
    # Define seasons and output path here or via environment/config
    seasons = ['2023-24', '2024-25']
    output_path = 'defensive_clusters.xlsx'

    pipeline = DefensiveShotClusterPipeline(
        seasons=seasons,
        output_path=output_path
    )
    pipeline.run()
