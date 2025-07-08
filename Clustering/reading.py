import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nba_api.stats.endpoints import leaguedashptdefend, leaguedashplayerstats
from nba_api.stats.static import players
from utils.base import _Base  # Metaclass providing logger injection

class DataFetcher(metaclass=_Base):
    """
    Fetch defensive shot data from NBA API in parallel.
    """
    def __init__(self, seasons: list):
        """
        Args:
            seasons (list): List of season strings to fetch.
        """
        self.seasons = seasons

    def _fetch_season(self, season: str):
        """
        Fetch stats for a single season.
        """
        self.logger.debug(f"Fetching data for season {season}")
        gen = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense='Defense',
            per_mode_detailed='PerGame'
        ).get_data_frames()[0].assign(SEASON=season)
        ls6 = leaguedashptdefend.LeagueDashPtDefend(
            season=season,
            season_type_all_star='Regular Season',
            defense_category='Less Than 6Ft'
        ).get_data_frames()[0].assign(SEASON=season)
        fp = leaguedashptdefend.LeagueDashPtDefend(
            season=season,
            season_type_all_star='Regular Season',
            defense_category='Less Than 10Ft'
        ).get_data_frames()[0].assign(SEASON=season)
        tp = leaguedashptdefend.LeagueDashPtDefend(
            season=season,
            season_type_all_star='Regular Season',
            defense_category='3 Pointers'
        ).get_data_frames()[0].assign(SEASON=season)
        return gen, ls6, fp, tp

    def fetch_all(self) -> tuple:
        """
        Fetch data for all seasons concurrently.

        Returns:
            tuple: (general_df, ls6_df, fp_df, tp_df, players_df)
        """
        self.logger.info("Starting parallel data fetch")
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._fetch_season, s): s for s in self.seasons}
            general_list, ls6_list, fp_list, tp_list = [], [], [], []
            for future in as_completed(futures):
                try:
                    gen, ls6, fp, tp = future.result()
                    general_list.append(gen)
                    ls6_list.append(ls6)
                    fp_list.append(fp)
                    tp_list.append(tp)
                except Exception as e:
                    self.logger.error(f"Error fetching season data: {e}")
                    raise
        general_df = pd.concat(general_list, ignore_index=True)
        ls6_df = pd.concat(ls6_list, ignore_index=True)
        fp_df = pd.concat(fp_list, ignore_index=True)
        tp_df = pd.concat(tp_list, ignore_index=True)
        players_df = pd.DataFrame(players.get_players()).rename(
            columns={'id': 'PLAYER_ID', 'full_name': 'PLAYER_NAME'}
        )
        self.logger.info("Data fetch completed")
        return general_df, ls6_df, fp_df, tp_df, players_df
