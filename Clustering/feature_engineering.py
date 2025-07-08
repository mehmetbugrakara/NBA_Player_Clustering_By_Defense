import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.base import _Base
import numpy as np  # Metaclass providing logger injection

class FeatureEngineer(metaclass=_Base):
    """
    Vectorized feature engineering for defensive metrics.
    """
    def __init__(self, recent_seasons: int = 2):
        """
        Args:
            recent_seasons (int): Number of recent seasons to average over.
        """
        self.recent_seasons = recent_seasons

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute averages, global differences, normalization, and defense score.

        Returns:
            pd.DataFrame: DataFrame with normalized features and defense_score.
        """
        self.logger.info("Starting feature engineering")
        try:
            # 2. Automatic defense metric selection
            meta_cols = [
                'PLAYER_ID','PLAYER_NAME','NICKNAME',
                'TEAM_ID','TEAM_ABBREVIATION','AGE',
                'GP','W','L','W_PCT','MIN',
                'SEASON','CLOSE_DEF_PERSON_ID',
                'PLAYER_POSITION_general','PLAYER_POSITION_from_point',
                'PLAYER_POSITION','Year','FGM_LT_10', 'FGA_LT_10','FG3M', 'FG3A','PCT_PLUSMINUS','PLUSMINUS_from_point','PLUSMINUS_three_point','FGM_LT_06', 'FGA_LT_06'
            ]
            metrics = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in meta_cols and not c.endswith('_RANK')
            ]

            # 3. Averages over the last 2 seasons
            latest_years = sorted(df['Year'].unique())[-2:]
            recent = df[df['Year'].isin(latest_years)]
            player_avg = (
                recent
                .groupby(['PLAYER_ID'])[metrics]
                .mean()
                .reset_index()
            )
            player_avg = player_avg.rename(columns={m: f"avg_{m}" for m in metrics})

            # 4. Add position information
            pos_map = (
                recent
                .groupby('PLAYER_ID')['PLAYER_POSITION']
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0])
                .rename('POSITION')
                .reset_index()
            )
            player_avg = player_avg.merge(pos_map, on='PLAYER_ID')

            # 5. League (global) statistics: mean/min/max on averages
            global_stats = {
                m: {
                    'mean': player_avg[f"avg_{m}"].mean(),
                    'min':  player_avg[f"avg_{m}"].min(),
                    'max':  player_avg[f"avg_{m}"].max()
                }
                for m in metrics
            }

            # 6. Position-based statistics: mean/min/max on averages
            avg_cols = [f"avg_{m}" for m in metrics]
            pos_stats = (
                player_avg
                .groupby('POSITION')[avg_cols]
                .agg(['mean','min','max'])
            )
            pos_stats.columns = [f"{col[0].split('avg_')[1]}_{col[1]}" for col in pos_stats.columns]
            pos_stats = pos_stats.reset_index()
            player_avg = player_avg.merge(pos_stats, on='POSITION', how='left')

            # 7. Create difference columns
            diff_cols = []
            for m in metrics:
                a = f"avg_{m}"
                # Global differences
                player_avg[f"diff_mean_global_{m}"] = player_avg[a] - global_stats[m]['mean']
                player_avg[f"diff_to_min_global_{m}"] = player_avg[a] - global_stats[m]['min']
                player_avg[f"diff_to_max_global_{m}"] = global_stats[m]['max'] - player_avg[a]
                # Position differences
                player_avg[f"diff_mean_pos_{m}"] = player_avg[a] - player_avg[f"{m}_mean"]
                player_avg[f"diff_to_min_pos_{m}"] = player_avg[a] - player_avg[f"{m}_min"]
                player_avg[f"diff_to_max_pos_{m}"] = player_avg[f"{m}_max"] - player_avg[a]
                diff_cols += [
                    f"diff_mean_global_{m}",
                    f"diff_to_min_global_{m}",
                    f"diff_to_max_global_{m}",
                    f"diff_mean_pos_{m}",
                    f"diff_to_min_pos_{m}",
                    f"diff_to_max_pos_{m}"
                ]

            # 8. Min–Max normalize each diff column using sklearn (0-1)
            scaler = MinMaxScaler()
            norm_values = scaler.fit_transform(player_avg[diff_cols])
            norm_col_names = [f"norm_{col}" for col in diff_cols]
            player_avg[norm_col_names] = norm_values

            # 9. Invert metrics where lower is better
            better_low = [
                'DEF_RATING','OPP_PTS_OFF_TOV','OPP_PTS_2ND_CHANCE',
                'OPP_PTS_FB','OPP_PTS_PAINT','D_FG_PCT','LT_10_PCT''NS_LT_10_PCT','D_FG_PCT', 'NORMAL_FG_PCT','NS_FG3_PCT','FGM_LT_10',
            ]
            for m in better_low:
                for diff in ['diff_mean_global','diff_to_min_global','diff_to_max_global',
                            'diff_mean_pos','diff_to_min_pos','diff_to_max_pos']:
                    col = f"norm_{diff}_{m}"
                    if col in player_avg:
                        player_avg[col] = 1 - player_avg[col]

            # 10. Final defense_score: average of all normalized columns
            player_avg['defense_score'] = player_avg[norm_col_names].mean(axis=1)
            player_avg.reset_index(inplace=True)

            self.logger.info("Feature engineering completed")
            return player_avg

        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}", exc_info=True)
            raise
