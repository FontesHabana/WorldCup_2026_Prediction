from dataclasses import dataclass, field
from typing import List
from functools import cmp_to_key

@dataclass
class TeamStats:
    name: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    was_penalty: bool = False

    @property
    def winner(self) -> str:
        """Calcula el ganador automáticamente basado en los goles."""
        if self.home_goals > self.away_goals:
            return self.home_team
        elif self.away_goals > self.home_goals:
            return self.away_team
        else:
            return "Draw"


@dataclass
class GroupResult:
    group_name: str
    standings: List[TeamStats]
    matches: List[MatchResult]

    def _get_h2h_match(self, team_a_name: str, team_b_name: str) -> MatchResult:
        """Busca el partido que jugaron entre sí dos equipos."""
        for m in self.matches:
            if (m.home_team == team_a_name and m.away_team == team_b_name) or \
                    (m.home_team == team_b_name and m.away_team == team_a_name):
                return m
        return None

    def compare_teams(self, a: TeamStats, b: TeamStats):
        """
        Lógica de comparación jerárquica (Puntos -> H2H -> General).
        """
        # 1. Puntos totales
        if a.points != b.points:
            return a.points - b.points

        # 2. Empate en puntos -> Mirar Head-to-Head (H2H)
        h2h = self._get_h2h_match(a.name, b.name)
        if h2h:
            # Puntos H2H
            pts_a = 3 if h2h.winner == a.name else (1 if h2h.winner == "Draw" else 0)
            pts_b = 3 if h2h.winner == b.name else (1 if h2h.winner == "Draw" else 0)
            if pts_a != pts_b:
                return pts_a - pts_b

            # DG H2H (Diferencia de goles en su partido)
            dg_h2h_a = h2h.home_goals - h2h.away_goals if h2h.home_team == a.name else h2h.away_goals - h2h.home_goals
            dg_h2h_b = -dg_h2h_a
            if dg_h2h_a != dg_h2h_b:
                return dg_h2h_a - dg_h2h_b

            # GF H2H (Goles a favor en su partido)
            gf_h2h_a = h2h.home_goals if h2h.home_team == a.name else h2h.away_goals
            gf_h2h_b = h2h.away_goals if h2h.home_team == a.name else h2h.home_goals
            if gf_h2h_a != gf_h2h_b:
                return gf_h2h_a - gf_h2h_b

        # 3. Si el H2H sigue empatado (fue un empate 1-1, por ejemplo),
        # pasamos a los criterios globales del grupo:

        # Diferencia de Goles TOTAL en el grupo
        if a.goal_difference != b.goal_difference:
            return a.goal_difference - b.goal_difference

        # Goles a favor TOTALES en el grupo
        if a.goals_for != b.goals_for:
            return a.goals_for - b.goals_for

        # 4. Si persiste el empate, podrías añadir sorteo o ranking FIFA
        return 0

    def sort_standings(self):
        """Ordena la tabla usando el comparador personalizado."""
        self.standings.sort(key=cmp_to_key(self.compare_teams), reverse=True)


@dataclass
class TournamentResult:
    tournament_name: str
    GroupResults: List[GroupResult]
    Round32:List[MatchResult]
    Round32Win:List[str]
    Round16:List[MatchResult]
    Round16Win:List[str]
    Round8:List[MatchResult]
    Round8Win:List[str]
    SemiFinal:List[MatchResult]
    SemiFinal:List[str]
    Final:MatchResult
    FinalWin:str