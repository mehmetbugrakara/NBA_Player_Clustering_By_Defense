import pandas as pd
from datetime import datetime
from utils.base import _Base  # Metaclass providing logger injection

class DataPreprocessor(metaclass=_Base):
    """
    Merge and filter raw defensive data.
    """
    def __init__(self, gp_threshold: int = 20, min_threshold: int = 12):
        """
        Args:
            gp_threshold (int): Minimum games played.
            min_threshold (int): Minimum minutes played.
        """
        self.gp_threshold = gp_threshold
        self.min_threshold = min_threshold

    def preprocess(self, general_df: pd.DataFrame, ls6_df: pd.DataFrame,
                   fp_df: pd.DataFrame, tp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge raw DataFrames and filter by thresholds.

        Returns:
            pd.DataFrame: Filtered and merged DataFrame.
        """
        self.logger.info("Starting preprocessing")
        try:
            ls6_sel = ls6_df[['CLOSE_DEF_PERSON_ID','PLAYER_NAME','PLAYER_POSITION',
                              'FGM_LT_06','FGA_LT_06','SEASON']].drop_duplicates()
            fp_sel = fp_df[['CLOSE_DEF_PERSON_ID','FGM_LT_10','FGA_LT_10','PLUSMINUS','SEASON']]
            tp_sel = tp_df[['CLOSE_DEF_PERSON_ID','FG3M','FG3A','PLUSMINUS','SEASON']]

            merged = ls6_sel.merge(fp_sel, on=['CLOSE_DEF_PERSON_ID','SEASON'])
            merged = merged.merge(tp_sel, on=['CLOSE_DEF_PERSON_ID','SEASON'])

            df = (
                general_df.merge(
                    merged,
                    left_on=['PLAYER_ID','SEASON'],
                    right_on=['CLOSE_DEF_PERSON_ID','SEASON'],
                    how='inner'
                )
                .query("GP > @self.gp_threshold and MIN > @self.min_threshold")
            )
            df['Year'] = df['SEASON'].apply(lambda s: self._convert_to_date(s).year)

            self.logger.info("Preprocessing completed")
            return df
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

    def _convert_to_date(self, season_str: str) -> datetime:
        """
        Convert season string to datetime at July 1 of the correct year.
        """
        if datetime.today().month <= 9:
            year = int('20' + season_str.split('-')[1])
        else:
            year = int(season_str.split('-')[0])
        return datetime(year, 7, 1)
