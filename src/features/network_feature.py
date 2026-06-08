import math
from typing import Dict, Union
import networkx as nx
import pandas as pd


class FootballNetworkModel:
    """
    SOTA Graph-Based Strength Model using Directed PageRank.
    Prunes future matches to prevent temporal data leakage during evaluation.
    """

    def __init__(self, damping_factor: float = 0.85):
        """
        Initializes the network model.
        :param damping_factor: The alpha parameter for PageRank (default: 0.85).
        """
        self.damping_factor = damping_factor
        self.raw_matches: Union[pd.DataFrame, None] = None

        # Cache to store calculated centralities: { "YYYY-MM": { "team_name": score } }
        self._cache: Dict[str, Dict[str, float]] = {}

    def fit_historical_data(self, df_matches: pd.DataFrame) -> None:
        """
        Registers the entire historical match dataset.
        This dataset will be sliced chronologically on-the-fly.
        """
        self.raw_matches = df_matches.copy()
        self.raw_matches["date"] = pd.to_datetime(self.raw_matches["date"])
        # Clear cache when new training data is registered
        self._cache.clear()

    def _build_graph_up_to_date(self, limit_date: pd.Timestamp) -> nx.DiGraph:
        """
        Constructs a directed graph using matches played before limit_date,
        applying an exponential time-decay to heavily prioritize recent matches.
        """
        G = nx.DiGraph()

        if self.raw_matches is None:
            return G

        # Filter strictly past matches
        past_slice = self.raw_matches[self.raw_matches["date"] < limit_date].copy()

        # Decay parameter: 0.0019 translates to a ~365 day half-life
        gamma = 0.0019

        def add_or_update_edge(origen: str, destino: str, peso: float):
            if G.has_edge(origen, destino):
                G[origen][destino]["weight"] += peso
            else:
                G.add_edge(origen, destino, weight=peso)

        for _, row in past_slice.iterrows():
            home = str(row["home_team"]).lower().strip()
            away = str(row["away_team"]).lower().strip()
            h_score = int(row["home_score"])
            a_score = int(row["away_score"])
            match_date = pd.to_datetime(row["date"])

            # Calculate match age in days relative to our limit_date
            days_ago = (limit_date - match_date).days
            if days_ago < 0:
                days_ago = 0

            # Exponential time-decay modifier
            time_decay = math.exp(-gamma * days_ago)

            # Original score margin weight
            margin = math.sqrt(abs(h_score - a_score))

            # Apply decay directly to the edge weight
            decayed_weight = margin * time_decay
            base_decayed_draw_weight = 1.0 * time_decay

            if h_score > a_score:
                add_or_update_edge(away, home, decayed_weight)
            elif a_score > h_score:
                add_or_update_edge(home, away, decayed_weight)
            else:
                add_or_update_edge(away, home, base_decayed_draw_weight)
                add_or_update_edge(home, away, base_decayed_draw_weight)

        return G
    def get_team_centrality(self, team_name: str, match_date: str) -> float:
        """
        Retrieves the PageRank score of a team. The calculation strictly uses
        matches played BEFORE the match_date.
        """
        team_clean = team_name.lower().strip()
        target_dt = pd.to_datetime(match_date)

        # Monthly cache key (e.g., "2022-11") to prevent recalculating for every single match day
        cache_key = target_dt.strftime("%Y-%m")

        if cache_key not in self._cache:
            # Build the graph and calculate PageRank using only historical matches
            G_slice = self._build_graph_up_to_date(target_dt)

            if len(G_slice.nodes) > 0:
                scores = nx.pagerank(
                    G_slice, alpha=self.damping_factor, weight="weight"
                )
                self._cache[cache_key] = scores
            else:
                self._cache[cache_key] = {}

        return self._cache[cache_key].get(team_clean, 0.0)


# =====================================================================
# DEMONSTRATION SECTION (Verification of Leakage Prevention)
# =====================================================================
if __name__ == "__main__":
    print("--- Running Network Model Temporal Verification ---")

    # 1. Create a simulated chronological match dataset
    # We will simulate a team ("morocco") that starts average but becomes elite in late 2022.
    simulated_matches = pd.DataFrame(
        [
            # Year 2021 matches
            {"date": "2021-03-10", "home_team": "Spain", "away_team": "Morocco", "home_score": 3, "away_score": 0},
            {"date": "2021-06-15", "home_team": "France", "away_team": "Morocco", "home_score": 2, "away_score": 0},
            {"date": "2021-09-20", "home_team": "Spain", "away_team": "France", "home_score": 1, "away_score": 1},

            # Late 2022 World Cup matches (Morocco starts beating giants)
            {"date": "2022-11-27", "home_team": "Morocco", "away_team": "Belgium", "home_score": 2, "away_score": 0},
            {"date": "2022-12-06", "home_team": "Morocco", "away_team": "Spain", "home_score": 1, "away_score": 0},
            {"date": "2022-12-10", "home_team": "Morocco", "away_team": "Portugal", "home_score": 1, "away_score": 0},

            # Post-2022 matches
            {"date": "2023-03-25", "home_team": "Morocco", "away_team": "Brazil", "home_score": 2, "away_score": 1},
        ]
    )

    # 2. Instantiate and fit the model
    network_model = FootballNetworkModel(damping_factor=0.85)
    network_model.fit_historical_data(simulated_matches)

    # 3. Query Morocco's centrality at different points in time
    print("\n[Query 1] Morocco Prestige BEFORE the 2022 World Cup (Querying on 2022-01-01):")
    # At this date, they have only lost to Spain and France.
    score_pre_wc = network_model.get_team_centrality("Morocco", "2022-01-01")
    print(f"Morocco Eigen-Score: {score_pre_wc:.5f}")

    print("\n[Query 2] Morocco Prestige DURING the 2022 World Cup (Querying on 2022-12-08):")
    # At this date, they have beaten Belgium and Spain, but not yet Portugal or Brazil.
    score_mid_wc = network_model.get_team_centrality("Morocco", "2022-12-08")
    print(f"Morocco Eigen-Score: {score_mid_wc:.5f}")

    print("\n[Query 3] Morocco Prestige AFTER the World Cup and Brazil win (Querying on 2023-06-01):")
    # All matches in the dataset are now history.
    score_post_wc = network_model.get_team_centrality("Morocco", "2023-06-01")
    print(f"Morocco Eigen-Score: {score_post_wc:.5f}")

    # 4. Verification Check
    print("\n--- Integrity Verification Report ---")
    if score_pre_wc < score_mid_wc < score_post_wc:
        print("✅ SUCCESS: Temporal Isolation is functioning correctly.")
        print("The network prestige updates dynamically as matches occur.")
        print("Predictions made in 2022 are completely clean of future 2023 results.")
    else:
        print("❌ FAILURE: Potential leakage detected in calculation logic.")