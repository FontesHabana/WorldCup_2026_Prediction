import streamlit as st
import pandas as pd
import json
import os
import glob
import re
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Mundial 2026 - Predicciones Dinámicas",
    page_icon="🏆",
    layout="wide"
)

# Intentar importar plotly para visualizaciones interactivas
try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Intentar importar el calculador de grupos real
try:
    from Group_Calculator import group_calculator

    HAS_GROUP_CALCULATOR = True
except ImportError:
    HAS_GROUP_CALCULATOR = False

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARDCODED_BASE_DIR = "/home/fontes/workspace/personal_projects/WorldCup_2026_Prediction"

if os.path.exists(HARDCODED_BASE_DIR):
    BASE_DIR = HARDCODED_BASE_DIR

RESULTS_DIR = os.path.join(BASE_DIR, "simulation_results")


# --- BUSCADORES INTELIGENTES DE ARCHIVOS CSV ---
def find_knockout_matches_csv():
    """Busca un archivo CSV que contenga la estructura del bracket de eliminación directa."""
    search_dirs = [
        RESULTS_DIR,
        BASE_DIR,
        os.getcwd(),
        os.path.join(os.getcwd(), "simulation_results")
    ]
    for s_dir in search_dirs:
        if not s_dir or not os.path.exists(s_dir):
            continue
        for file in os.listdir(s_dir):
            if file.endswith(".csv"):
                full_path = os.path.join(s_dir, file)
                try:
                    df_preview = pd.read_csv(full_path, nrows=3)
                    # Verifica columnas características del archivo de eliminación directa
                    if "Match_ID" in df_preview.columns and "Predicted_Winner_Overall" in df_preview.columns:
                        return full_path
                except Exception:
                    continue
    return None


def find_probabilities_csv():
    """Busca un archivo CSV que contenga las probabilidades de avance por fase."""
    # Buscar primero específicamente el archivo direct_knockout_progression.csv
    exact_candidate = find_file_by_name("direct_knockout_progression.csv")
    if exact_candidate and os.path.exists(exact_candidate):
        return exact_candidate

    search_dirs = [
        RESULTS_DIR,
        BASE_DIR,
        os.getcwd(),
        os.path.join(os.getcwd(), "simulation_results")
    ]
    for s_dir in search_dirs:
        if not s_dir or not os.path.exists(s_dir):
            continue
        for file in os.listdir(s_dir):
            if file.endswith(".csv"):
                full_path = os.path.join(s_dir, file)
                try:
                    df_preview = pd.read_csv(full_path, nrows=3)
                    cols_lower = [c.lower() for c in df_preview.columns]
                    if "team" in cols_lower and any(
                            "champion" in c or "semi-final" in c or "finalist" in c for c in cols_lower):
                        return full_path
                except Exception:
                    continue
    return None


def find_file_by_name(filename_pattern):
    """Busca un archivo por coincidencia de nombre en los directorios de trabajo."""
    search_dirs = [
        RESULTS_DIR,
        BASE_DIR,
        os.getcwd(),
        os.path.join(os.getcwd(), "simulation_results")
    ]
    for s_dir in search_dirs:
        if not s_dir or not os.path.exists(s_dir):
            continue
        exact_path = os.path.join(s_dir, filename_pattern)
        if os.path.exists(exact_path):
            return exact_path
        glob_pattern = os.path.join(s_dir, f"*{filename_pattern}*")
        candidates = glob.glob(glob_pattern)
        if candidates:
            return candidates[0]
    return None


# Ubicación dinámica de los orígenes de datos
PATH_KO_CSV = find_knockout_matches_csv()
PATH_FASES_KO = find_probabilities_csv()
PATH_GRUPOS = find_file_by_name("probabilidades_grupos.csv")
PATH_MATCHES = find_file_by_name("partidos_mas_probables_por_grupo.csv")
PATH_GROUP_FINAL_RESULTS = find_file_by_name("wc2026_group_stage.csv")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .titulo-principal { color: #1b5e20; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; text-align: center; margin-bottom: 5px; }
    .sub-titulo { color: #555555; text-align: center; font-size: 1.1rem; margin-bottom: 25px; }
    .stDataFrame { border: 1px solid #e0e0e0; border-radius: 5px; }
    .bracket-match { border: 1px solid #ddd; padding: 10px; border-radius: 8px; margin-bottom: 10px; background-color: #f9f9f9; }
    .match-winner { font-weight: bold; color: #2e7d32; }
    .group-header { background-color: #1b5e20; color: white; padding: 5px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='titulo-principal'>🏆 Copa Mundial de la FIFA 2026</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-titulo'>Análisis probabilístico y proyecciones de simulación</p>", unsafe_allow_html=True)


# --- CARGA DINÁMICA DE METADATA ---
@st.cache_data
def get_dynamic_metadata():
    if not PATH_MATCHES or not os.path.exists(PATH_MATCHES):
        return {}, {}

    try:
        df = pd.read_csv(PATH_MATCHES)
        mapping = {}
        grupos = {}

        for _, row in df.iterrows():
            g = f"Grupo {row['Group']}"
            t1, t2 = str(row['Team_A']).lower(), str(row['Team_B']).lower()

            mapping[t1] = t1.replace('_', ' ').title()
            mapping[t2] = t2.replace('_', ' ').title()

            if g not in grupos:
                grupos[g] = set()
            grupos[g].add(t1)
            grupos[g].add(t2)

        return mapping, {k: sorted(list(v)) for k, v in grupos.items()}
    except Exception:
        return {}, {}


TEAM_MAPPING_DYNAMIC, GRUPOS_TEAMS_DYNAMIC = get_dynamic_metadata()


def get_display_name(slug):
    if not slug or pd.isna(slug):
        return "Por determinar"
    slug_clean = str(slug).lower().strip()
    return TEAM_MAPPING_DYNAMIC.get(slug_clean, slug_clean.replace('_', ' ').title())


# --- CARGA Y PROCESAMIENTO DE DATOS ---
@st.cache_data
def load_data():
    df_fases = pd.DataFrame()
    if PATH_FASES_KO and os.path.exists(PATH_FASES_KO):
        df_fases = pd.read_csv(PATH_FASES_KO)
        df_fases.columns = [c.replace(' (%)', '').strip() for c in df_fases.columns]
        if not df_fases.empty and (df_fases.columns[0] == "Unnamed: 0" or "Team" not in df_fases.columns):
            df_fases = df_fases.rename(columns={df_fases.columns[0]: "Team"})

    df_grupos = pd.read_csv(PATH_GRUPOS) if PATH_GRUPOS and os.path.exists(PATH_GRUPOS) else pd.DataFrame()
    df_matches = pd.read_csv(PATH_MATCHES) if PATH_MATCHES and os.path.exists(PATH_MATCHES) else pd.DataFrame()
    df_grupos_resultados = pd.read_csv(PATH_GROUP_FINAL_RESULTS) if PATH_GROUP_FINAL_RESULTS and os.path.exists(
        PATH_GROUP_FINAL_RESULTS) else pd.DataFrame()

    # Construcción dinámica del bracket a partir del CSV de eliminación directa
    bracket = {}
    if PATH_KO_CSV and os.path.exists(PATH_KO_CSV):
        try:
            df_ko = pd.read_csv(PATH_KO_CSV)
            for _, row in df_ko.iterrows():
                m_id = str(row['Match_ID']).strip()
                bracket[m_id] = {
                    'predicted_winner': row.get('Predicted_Winner_Overall', ''),
                    'predicted_score': row.get('Most_Probable_Score', '?-?'),
                    'matchup': f"{row.get('Team_A', 'TBD')} vs {row.get('Team_B', 'TBD')}",
                    'winner_probability_pct': float(row.get('Winner_Prob_Within_Matchup_Pct', 0))
                }
        except Exception as e:
            st.error(f"Error al procesar el archivo de cruces directos: {e}")

    return df_fases, df_grupos, bracket, df_matches, df_grupos_resultados


df_fases, df_grupos, bracket_data, df_matches_all, df_grupos_resultados = load_data()

# Standings reales en caso de estar configurados
df_standings = pd.DataFrame()
if HAS_GROUP_CALCULATOR and not df_grupos_resultados.empty:
    try:
        df_standings = group_calculator(df_grupos_resultados)
    except Exception:
        df_standings = pd.DataFrame()


# --- FUNCIONES DE COMPARACIÓN GLOBAL DE RESULTADOS ---
def find_real_score_exact(team_a_pred, team_b_pred, df_real):
    if df_real is None or df_real.empty:
        return None

    # Normalización de nombres de equipos
    def normalize(name):
        if not name or pd.isna(name):
            return ""
        t = str(name).lower().strip().replace('_', ' ')
        equivalencias = {
            'czechia': 'czech republic',
            'türkiye': 'turkey',
            'usa': 'united states',
        }
        return equivalencias.get(t, t)

    ta_target = normalize(team_a_pred)
    tb_target = normalize(team_b_pred)

    for _, row in df_real.iterrows():
        if 'home_name' not in row or 'away_name' not in row or 'result' not in row:
            continue

        r_ta = normalize(row['home_name'])
        r_tb = normalize(row['away_name'])
        raw_result = str(row['result']).strip()

        if not raw_result or '-' not in raw_result:
            continue

        # Coincidencia directa
        if r_ta == ta_target and r_tb == tb_target:
            try:
                parts = raw_result.split('-')
                return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, IndexError):
                pass
        # Coincidencia invertida
        elif r_ta == tb_target and r_tb == ta_target:
            try:
                parts = raw_result.split('-')
                # Se invierte el orden para alinearlo con el equipo A proyectado
                return int(parts[1].strip()), int(parts[0].strip())
            except (ValueError, IndexError):
                pass
    return None


# --- PESTAÑAS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Probabilidades",
    "📅 Fase de Grupos",
    "🏆 Bracket KO",
    "🎯 Efectividad"
])

# --- TAB 1: PROBABILIDADES ---
with tab1:
    st.subheader("Análisis Estadístico y Probabilidades del Torneo")

    if not df_fases.empty:
        # Función interna para ordenar lógicamente las columnas de fases para los gráficos
        def get_sorted_stages(columns):
            ordered_keywords = [
                'group', 'grupo',
                '32', '1/16', 'dieciseis',
                '16', 'octavos', 'octavo',
                '8', 'cuartos', 'quarter',
                '4', 'semi', 'semifinal',
                'finalist', 'finalista',
                'final',
                'champion', 'campeon', 'winner', 'titulo'
            ]
            stages = [col for col in columns if col.lower() != 'team']

            def stage_rank(stage_name):
                name_lower = stage_name.lower()
                for rank, keyword in enumerate(ordered_keywords):
                    if keyword in name_lower:
                        return rank
                return 999

            return sorted(stages, key=stage_rank)


        # Determinar columna de campeón para los favoritos
        col_c = next(
            (col for col in df_fases.columns if any(kw in col.lower() for kw in ['champion', 'campeon', 'winner'])), "")
        if not col_c and len(df_fases.columns) > 1:
            col_c = df_fases.columns[-1]

        if col_c and col_c in df_fases.columns:
            st.markdown("### Selecciones con mayor probabilidad de título")
            df_favoritos = df_fases.sort_values(by=col_c, ascending=False).head(4)
            cols_fav = st.columns(4)
            for idx, (_, row) in enumerate(df_favoritos.iterrows()):
                with cols_fav[idx]:
                    st.metric(
                        label=f"{idx + 1}º Favorito: {get_display_name(row['Team'])}",
                        value=f"{row[col_c]:.2f}%"
                    )

        st.write("---")

        st.markdown("### Buscador de Trayectoria por Selección")
        team_search = st.selectbox(
            "Selecciona un país para ver sus probabilidades de avance detalladas:",
            sorted(df_fases["Team"].unique())
        )

        if team_search:
            team_data = df_fases[df_fases["Team"] == team_search].iloc[0]

            # Se extraen las fases respetando el orden secuencial natural del archivo de origen (cronológico)
            fases_labels = [col for col in df_fases.columns if col.lower() not in ['team', 'unnamed: 0']]
            valores_fases = [float(team_data[col]) for col in fases_labels]

            # Etiquetas con formato de título para la presentación visual
            x_labels_ordenados = [col.replace('_', ' ').title() for col in fases_labels]

            col_search_chart, col_search_data = st.columns([2, 1])
            with col_search_chart:
                if HAS_PLOTLY:
                    # Estructurar los datos en un DataFrame temporal para garantizar el comportamiento de Plotly
                    df_temp_plot = pd.DataFrame({
                        "Fase": x_labels_ordenados,
                        "Probabilidad": valores_fases
                    })

                    fig_search = px.line(
                        df_temp_plot,
                        x="Fase",
                        y="Probabilidad",
                        markers=True,
                        title=f"Probabilidad de supervivencia de {get_display_name(team_search)} por fase",
                        labels={"Fase": "Fase del Torneo", "Probabilidad": "Probabilidad de avance (%)"},
                        category_orders={"Fase": x_labels_ordenados}
                        # Fuerza la visualización exacta en orden cronológico
                    )

                    fig_search.update_layout(
                        yaxis_range=[-5, 105],
                        height=350
                    )
                    st.plotly_chart(fig_search, use_container_width=True)
                else:
                    # Fallback usando el gráfico de área nativo de Streamlit
                    df_chart_fallback = pd.DataFrame(
                        {"Probabilidad de avance (%)": valores_fases},
                        index=x_labels_ordenados
                    )
                    st.area_chart(df_chart_fallback)

            with col_search_data:
                df_resumen_individual = pd.DataFrame({
                    "Fase": x_labels_ordenados,
                    "Probabilidad": [f"{val:.2f}%" for val in valores_fases]
                })
                st.dataframe(df_resumen_individual, use_container_width=True, hide_index=True)

        st.write("---")



        st.markdown("### Matriz Completa de Datos de Simulación")
        display_df = df_fases.copy()
        display_df['Team'] = display_df['Team'].apply(get_display_name)
        display_df = display_df.set_index('Team')

        # Ordenar columnas de la matriz en orden lógico del torneo
        sorted_cols = get_sorted_stages(display_df.columns)
        display_df = display_df[sorted_cols]

        st.dataframe(
            display_df.style.background_gradient(cmap='YlGn', axis=None).format("{:.2f}%"),
            height=500,
            use_container_width=True
        )
    else:
        st.warning("Los datos del modelo de probabilidades no se encuentran disponibles.")
# --- TAB 2: FASE DE GRUPOS ---
with tab2:
    if GRUPOS_TEAMS_DYNAMIC:
        grupo_sel = st.selectbox("Selecciona un Grupo:", sorted(list(GRUPOS_TEAMS_DYNAMIC.keys())))
        g_id = grupo_sel.split(" ")[1]

        c1, c2 = st.columns([1, 1.2])

        with c1:
            st.markdown(f"### Tabla Proyectada {grupo_sel}")
            tabla_path = find_file_by_name(f"tabla_mas_probable_{grupo_sel.lower().replace(' ', '_')}.csv")
            if tabla_path and os.path.exists(tabla_path):
                df_t = pd.read_csv(tabla_path)
                df_t['Team'] = df_t['Team'].apply(get_display_name)
                st.table(df_t.set_index('Pos'))

            st.markdown("### Probabilidades de Posición")
            if not df_grupos.empty:
                equipos_g = GRUPOS_TEAMS_DYNAMIC.get(grupo_sel, [])
                df_g_prob = df_grupos[df_grupos['Team'].isin(equipos_g)].copy()
                df_g_prob['Team'] = df_g_prob['Team'].apply(get_display_name)
                columnas_posiciones = [c for c in df_g_prob.columns if c != "Team"]
                st.dataframe(
                    df_g_prob.set_index('Team')[columnas_posiciones].style.background_gradient(cmap='Blues',
                                                                                               axis=1).format(
                        "{:.1f}%"),
                    use_container_width=True
                )

            st.markdown(f"### Tabla de Posiciones Real ({grupo_sel})")
            if not df_standings.empty:
                group_table = df_standings[df_standings['group'] == g_id].copy()

                if not group_table.empty:
                    display_cols = {
                        'Pos': 'Pos',
                        'team_emoji': 'Bnd',
                        'team_name': 'Equipo',
                        'Pld': 'PJ',
                        'Pts': 'Pts',
                        'W': 'G',
                        'D': 'E',
                        'L': 'P',
                        'GF': 'GF',
                        'GA': 'GC',
                        'GD': 'DG',
                    }
                    available_cols = [col for col in display_cols.keys() if col in group_table.columns]
                    group_table_filtered = group_table[available_cols].rename(columns=display_cols)
                    st.dataframe(
                        group_table_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Pos": st.column_config.NumberColumn(width="small"),
                            "Bnd": st.column_config.TextColumn(width="small"),
                            "Equipo": st.column_config.TextColumn(width="medium"),
                            "Pts": st.column_config.NumberColumn(format="%d")
                        }
                    )
                else:
                    st.info("Aún no se registran enfrentamientos disputados en este grupo.")
            else:
                st.info("Las estadísticas reales se cargarán dinámicamente con el inicio de la competición.")

        with c2:
            st.markdown(f"### Partidos Proyectados {grupo_sel}")
            if not df_matches_all.empty:
                df_m_g = df_matches_all[df_matches_all['Group'] == g_id]
                if not df_m_g.empty:
                    for _, row in df_m_g.iterrows():
                        team_a_slug = row['Team_A']
                        team_b_slug = row['Team_B']
                        proj_g_a = int(row['Goals_A'])
                        proj_g_b = int(row['Goals_B'])

                        # Buscar el marcador real correspondiente
                        real_score = find_real_score_exact(team_a_slug, team_b_slug, df_grupos_resultados)

                        if real_score is not None:
                            real_g_a, real_g_b = real_score

                            # Calcular la diferencia de goles de ambos casos para validar acierto de signo (1, X, 2)
                            proj_diff = proj_g_a - proj_g_b
                            real_diff = real_g_a - real_g_b

                            is_correct = (proj_diff > 0 and real_diff > 0) or \
                                         (proj_diff < 0 and real_diff < 0) or \
                                         (proj_diff == 0 and real_diff == 0)

                            if is_correct:
                                bg_color = "#e8f5e9"  # Verde para aciertos
                                border_color = "#2e7d32"
                                status_label = "🎯 Acierto"
                            else:
                                bg_color = "#ffebee"  # Rojo para fallos
                                border_color = "#c62828"
                                status_label = "❌ Fallo"

                            st.markdown(f"""
                            <div style='border: 1px solid {border_color}; padding: 10px; border-radius: 8px; margin-bottom: 8px; background-color: {bg_color};'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <span style='font-weight: 600; width: 35%; text-align: left;'>{get_display_name(team_a_slug)}</span>
                                    <div style='width: 30%; text-align: center;'>
                                        <div style='font-size: 0.72rem; color: #555;'>Proyectado: <b>{proj_g_a} - {proj_g_b}</b></div>
                                        <div style='font-size: 1.05rem; font-weight: bold; margin: 2px 0; color: #111;'>Real: {real_g_a} - {real_g_b}</div>
                                    </div>
                                    <span style='font-weight: 600; width: 35%; text-align: right;'>{get_display_name(team_b_slug)}</span>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 5px; font-size: 0.75rem; color: #555; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 4px;'>
                                    <span>Confianza: {row['Score_Probability_Pct']:.1f}%</span>
                                    <span style='color: {border_color}; font-weight: bold;'>{status_label}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Renderizado normal si el partido no posee un resultado real en el dataset
                            res_color = "#e8f5e9" if row.get('Is_Group_Most_Probable', False) else "white"
                            st.markdown(f"""
                            <div style='border:1px solid #ddd; padding:8px; border-radius:5px; margin-bottom:5px; background-color:{res_color}'>
                                <div style='display:flex; justify-content:space-between;'>
                                    <span>{get_display_name(team_a_slug)}</span>
                                    <b>{proj_g_a} - {proj_g_b}</b>
                                    <span>{get_display_name(team_b_slug)}</span>
                                </div>
                                <div style='font-size:0.8rem; color:grey; text-align:center;'>Confianza: {row['Score_Probability_Pct']:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.write("No hay proyecciones disponibles para este grupo.")
            else:
                st.warning("Los datos dinámicos del torneo no pudieron ser leídos correctamente.")
# --- TAB 3: BRACKET ---
with tab3:
    st.subheader("Análisis de Fase de Eliminación Directa")

    # --- 1. CARGA DE RESULTADOS REALES DE ELIMINACIÓN DIRECTA ---
    PATH_REAL_KO = find_file_by_name("wc2026_knockout_stage.csv")
    df_real_ko = pd.read_csv(PATH_REAL_KO) if PATH_REAL_KO and os.path.exists(PATH_REAL_KO) else pd.DataFrame()


    # --- 2. GENERADOR DE BRACKETS EN HTML (DISEÑO LIMPIO) ---
    def generate_bracket_html(bracket_data, df_real=None, is_real_comparison=False):
        """Genera el documento HTML del Bracket para proyecciones o comparaciones reales."""

        real_matches = {}
        if df_real is not None and not df_real.empty:
            for _, r in df_real.iterrows():
                m_id = str(r.get('Match_ID', '')).strip()
                if m_id:
                    real_matches[m_id] = {
                        'team_a': r.get('Team_A', 'TBD'),
                        'team_b': r.get('Team_B', 'TBD'),
                        'goals_a': r.get('Goals_A', None),
                        'goals_b': r.get('Goals_B', None),
                        'winner': str(r.get('Winner', '')).strip().lower()
                    }

        def safe_int(val):
            if val is None or pd.isna(val):
                return None
            s = str(val).strip().lower()
            if s in ('', 'none', '<null>', 'null', 'nan', '?'):
                return None
            try:
                return int(float(s))
            except ValueError:
                return None

        def get_match_html(m_id, grid_row=None):
            m_proj = bracket_data.get(m_id)
            row_style = f"style='grid-row: {grid_row} !important;'" if grid_row else ""

            if not m_proj:
                return f"""
                <div class="bracket-match-card pending-card" {row_style}>
                    <div class="match-id">{m_id.replace('M', 'Partido ')}</div>
                    <div class="team-row" style="color: #94a3b8; font-style: italic;">Por definir</div>
                    <div class="team-row" style="color: #94a3b8; font-style: italic;">Por definir</div>
                    <div class="score-row">-</div>
                </div>
                """

            proj_winner = str(m_proj.get('predicted_winner', '')).strip().lower()
            proj_score = m_proj.get('predicted_score', '?-?')
            matchup = m_proj.get('matchup', 'TBD vs TBD')
            prob = m_proj.get('winner_probability_pct', 0)

            teams_proj = matchup.split(' vs ')
            t_a_proj_slug = teams_proj[0].strip().lower() if len(teams_proj) > 0 else "tbd"
            t_b_proj_slug = teams_proj[1].strip().lower() if len(teams_proj) > 1 else "tbd"

            nums = re.findall(r'\d+', proj_score)
            proj_score_display = f"{nums[0]} - {nums[1]}" if len(nums) >= 2 else proj_score

            card_class = ""
            badge_html = ""
            display_team_a = get_display_name(t_a_proj_slug)
            display_team_b = get_display_name(t_b_proj_slug)
            score_line_display = f"Proyectado: {proj_score_display}"
            winner_display = get_display_name(proj_winner) if proj_winner else "TBD"

            if is_real_comparison:
                m_real = real_matches.get(m_id)
                goals_a_parsed = safe_int(m_real.get('goals_a')) if m_real else None
                goals_b_parsed = safe_int(m_real.get('goals_b')) if m_real else None

                if m_real and goals_a_parsed is not None and goals_b_parsed is not None:
                    t_a_real_slug = str(m_real['team_a']).strip().lower()
                    t_b_real_slug = str(m_real['team_b']).strip().lower()
                    display_team_a = get_display_name(t_a_real_slug)
                    display_team_b = get_display_name(t_b_real_slug)

                    real_goals_a = goals_a_parsed
                    real_goals_b = goals_b_parsed
                    real_winner = m_real['winner']

                    score_line_display = f"Real: {real_goals_a} - {real_goals_b} (Proj: {proj_score_display})"
                    winner_display = f"Real: {get_display_name(real_winner)}"

                    if real_winner == proj_winner:
                        card_class = "correct-card"
                        badge_html = "<div class='badge-outcome badge-success'>🎯 ACIERTO</div>"
                    else:
                        card_class = "failed-card"
                        badge_html = f"<div class='badge-outcome badge-fail'>❌ FALLO (Proj: {get_display_name(proj_winner)})</div>"
                else:
                    card_class = "pending-card"
                    score_line_display = f"Proj: {proj_score_display}"
                    badge_html = "<div class='badge-outcome badge-pending'>Pendiente</div>"
            else:
                card_class = "normal-card"
                badge_html = f"<div class='winner-row'>🏆 {winner_display} ({prob:.1f}%)</div>"

            style_a = "font-weight: bold; color: #1b5e20;" if proj_winner == t_a_proj_slug else ""
            style_b = "font-weight: bold; color: #1b5e20;" if proj_winner == t_b_proj_slug else ""

            return f"""
            <div class="bracket-match-card {card_class}" {row_style}>
                <div class="match-id">{m_id.replace('M', 'Partido ')}</div>
                <div class="team-row" style="{style_a}">
                    <span>⚽ {display_team_a}</span>
                </div>
                <div class="team-row" style="{style_b}">
                    <span>⚽ {display_team_b}</span>
                </div>
                <div class="score-row"><b>{score_line_display}</b></div>
                {badge_html}
            </div>
            """

        r32_rows = ["1 / 4", "5 / 8", "9 / 12", "13 / 16", "17 / 20", "21 / 24", "25 / 28", "29 / 32"]
        r16_rows = ["3 / 6", "11 / 14", "19 / 22", "27 / 30"]
        r8_rows = ["7 / 10", "23 / 26"]
        r4_rows = ["15 / 18"]

        col1_html = "".join([get_match_html(f"M{73 + i}", r32_rows[i]) for i in range(8)])
        col2_html = "".join([get_match_html(f"M{89 + i}", r16_rows[i]) for i in range(4)])
        col3_html = "".join([get_match_html(f"M{97 + i}", r8_rows[i]) for i in range(2)])
        col4_html = get_match_html("M101", r4_rows[0])

        final_match_html = get_match_html("M104", "11 / 14")

        m_final = bracket_data.get("M104", {})
        champion_slug = m_final.get('predicted_winner', '')
        champ_name = get_display_name(champion_slug) if champion_slug else "Por determinar"

        if is_real_comparison and df_real is not None and not df_real.empty:
            f_real = df_real[df_real['Match_ID'] == 'M104']
            if not f_real.empty and pd.notna(f_real.iloc[0].get('Winner')):
                real_champ = str(f_real.iloc[0]['Winner']).strip().lower()
                champ_name = f"Real: {get_display_name(real_champ)}"
                champ_class = "correct-champ" if real_champ == champion_slug.lower() else "failed-champ"
            else:
                champ_class = "pending-champ"
        else:
            champ_class = "normal-champ"

        champ_box_html = ""
        if champion_slug or is_real_comparison:
            champ_box_html = f"""
            <div class="bracket-match-card champion-box {champ_class}" style="grid-row: 15 / 19 !important;">
                <div class="match-id champion-title" style="border-bottom-color: #f59e0b !important;">🏆 CAMPEÓN</div>
                <div style="font-size: 1.05rem; font-weight: bold; text-align: center; color: #b45309; padding: 4px 0;">
                    {champ_name}
                </div>
            </div>
            """

        m103_label_html = ""
        m103_html = ""
        if "M103" in bracket_data:
            m103_label_html = '<div class="tercer-puesto-label">TERCER PUESTO</div>'
            m103_html = get_match_html("M103", "21 / 24")

        col6_html = get_match_html("M102", r4_rows[0])
        col7_html = "".join([get_match_html(f"M{99 + i}", r8_rows[i]) for i in range(2)])
        col8_html = "".join([get_match_html(f"M{93 + i}", r16_rows[i]) for i in range(4)])
        col9_html = "".join([get_match_html(f"M{81 + i}", r32_rows[i]) for i in range(8)])

        return f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ margin: 0; padding: 10px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: transparent; }}
                .bracket-outer-container {{ width: 100%; overflow-x: auto; display: block; background-color: #f8fafc; border-radius: 12px; padding: 15px; border: 1px solid #e2e8f0; box-sizing: border-box; }}
                .bracket-wrapper {{ display: inline-flex; flex-direction: row; gap: 20px; padding: 10px 5px; }}
                .bracket-column {{ display: inline-flex; flex-direction: column; width: 180px; min-width: 180px; max-width: 180px; flex-shrink: 0; }}
                .bracket-header {{ font-size: 0.75rem; font-weight: bold; color: #ffffff; background-color: #1b5e20; text-align: center; padding: 8px 4px; border-radius: 6px; margin-bottom: 15px; text-transform: uppercase; height: 16px; line-height: 16px; }}
                .bracket-grid {{ display: grid; grid-template-rows: repeat(32, 35px); height: 1120px; width: 100%; position: relative; }}

                .bracket-match-card {{ background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 6px 8px; font-size: 0.72rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04); display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box; height: 100%; margin: 0; }}

                .normal-card {{ border-left: 4px solid #1b5e20; }}
                .correct-card {{ border: 1.5px solid #2e7d32; border-left: 6px solid #2e7d32; background-color: #e8f5e9; }}
                .failed-card {{ border: 1.5px solid #c62828; border-left: 6px solid #c62828; background-color: #ffebee; }}
                .pending-card {{ border-left: 4px solid #64748b; background-color: #f8fafc; }}

                .badge-outcome {{ font-size: 0.65rem; font-weight: bold; text-align: center; border-radius: 4px; padding: 1px; margin-top: 2px; }}
                .badge-success {{ background-color: #c8e6c9; color: #2e7d32; }}
                .badge-fail {{ background-color: #ffcdd2; color: #c62828; }}
                .badge-pending {{ background-color: #e2e8f0; color: #475569; }}

                .champion-box {{ background: linear-gradient(135deg, #ffffff 0%, #fef3c7 100%) !important; border: 2px solid #f59e0b !important; border-left: 6px solid #f59e0b !important; }}
                .correct-champ {{ background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%) !important; border: 2px solid #2e7d32 !important; border-left: 6px solid #2e7d32 !important; }}
                .failed-champ {{ background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%) !important; border: 2px solid #c62828 !important; border-left: 6px solid #c62828 !important; }}

                .match-id {{ font-size: 0.65rem; color: #64748b; font-weight: 600; border-bottom: 1px solid #f1f5f9; padding-bottom: 1px; margin-bottom: 1px; }}
                .team-row {{ display: flex; justify-content: space-between; align-items: center; padding: 1px 0; font-size: 0.72rem; color: #1e293b; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                .score-row {{ font-size: 0.68rem; color: #475569; background-color: #f8fafc; padding: 2px; border-radius: 4px; text-align: center; margin-top: 1px; border: 1px solid #e2e8f0; }}
                .winner-row {{ font-size: 0.68rem; font-weight: bold; color: #166534; text-align: center; margin-top: 1px; background-color: #f0fdf4; padding: 1px; border-radius: 4px; }}
                .champion-title {{ color: #b45309 !important; font-weight: bold; font-size: 0.75rem !important; }}
                .tercer-puesto-label {{ grid-row: 20 / 21; font-size: 0.68rem; font-weight: bold; color: #475569; text-align: center; align-self: end; }}
            </style>
        </head>
        <body>
            <div class="bracket-outer-container">
                <div class="bracket-wrapper">
                    <div class="bracket-column">
                        <div class="bracket-header">1/16 Final</div>
                        <div class="bracket-grid">{col1_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Octavos</div>
                        <div class="bracket-grid">{col2_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Cuartos</div>
                        <div class="bracket-grid">{col3_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Semifinal</div>
                        <div class="bracket-grid">{col4_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header" style="background-color: #d97706;">FINAL</div>
                        <div class="bracket-grid">
                            {final_match_html}
                            {champ_box_html}
                            {m103_label_html}
                            {m103_html}
                        </div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Semifinal</div>
                        <div class="bracket-grid">{col6_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Cuartos</div>
                        <div class="bracket-grid">{col7_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">Octavos</div>
                        <div class="bracket-grid">{col8_html}</div>
                    </div>
                    <div class="bracket-column">
                        <div class="bracket-header">1/16 Final</div>
                        <div class="bracket-grid">{col9_html}</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """


    # --- 3. RENDERIZADO DE BRACKETS ---
    st.write("---")
    st.markdown("### 🌳 Bracket Proyectado Original")
    html_proyeccion = generate_bracket_html(bracket_data, is_real_comparison=False)
    st.components.v1.html(html_proyeccion, height=1210, scrolling=True)

    st.write("---")
    st.markdown("### 🔄 Comparativa: Proyección vs Resultados Reales")
    if not df_real_ko.empty:
        html_comparativo = generate_bracket_html(bracket_data, df_real=df_real_ko, is_real_comparison=True)
        st.components.v1.html(html_comparativo, height=1210, scrolling=True)
    else:
        st.info(
            "Para visualizar el bracket comparativo, inicialice el archivo 'wc2026_knockout_stage.csv' con los resultados reales.")

    # --- 4. BUSCADOR Y MATRIZ DE CALOR (AL FINAL DE LA PÁGINA) ---
    st.write("---")
    st.markdown("### 📊 Matriz de Probabilidades por Partido")


    @st.cache_data
    def load_goal_probabilities():
        path = find_file_by_name("direct_knockout_goal_probabilities.csv")
        if not path:
            path = find_file_by_name("goal_probabilities")
        if path and os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()


    df_goal_probs = load_goal_probabilities()

    if not df_goal_probs.empty:
        match_metadata = df_goal_probs.groupby('Match_ID').first().reset_index()

        options_map = {}
        for _, r in match_metadata.iterrows():
            m_id = r['Match_ID']
            teams = str(r['Matchup']).split(' vs ')
            team_a_clean = get_display_name(teams[0]) if len(teams) > 0 else "TBD"
            team_b_clean = get_display_name(teams[1]) if len(teams) > 1 else "TBD"
            label = f"{m_id} ({r['Stage']}): {team_a_clean} vs {team_b_clean}"
            options_map[label] = m_id

        selected_label = st.selectbox(
            "Selecciona un partido para analizar la distribución de probabilidad de goles (Matriz de Calor):",
            list(options_map.keys())
        )

        if selected_label:
            target_id = options_map[selected_label]
            df_match_probs = df_goal_probs[df_goal_probs['Match_ID'] == target_id].copy()

            # Obtener metadatos del partido seleccionado
            selected_row = match_metadata[match_metadata['Match_ID'] == target_id].iloc[0]
            teams_parsed = str(selected_row['Matchup']).split(' vs ')
            team_a_label = get_display_name(teams_parsed[0]) if len(teams_parsed) > 0 else "Team A"
            team_b_label = get_display_name(teams_parsed[1]) if len(teams_parsed) > 1 else "Team B"

            # Construir la matriz de calor en formato 6x6 (goles de 0 a 5)
            matrix_size = 6
            matrix_data = [[0.0] * matrix_size for _ in range(matrix_size)]

            for _, row in df_match_probs.iterrows():
                score_line = str(row['Score_Line']).strip()
                prob = float(row['Probability_Pct'])
                nums = re.findall(r'\d+', score_line)
                if len(nums) >= 2:
                    g_a = int(nums[0])
                    g_b = int(nums[1])
                    if g_a < matrix_size and g_b < matrix_size:
                        matrix_data[g_a][g_b] = prob

            # Crear DataFrame con índices correspondientes
            df_matrix = pd.DataFrame(
                matrix_data,
                index=[f"Goles {team_a_label}: {i}" for i in range(matrix_size)],
                columns=[f"Goles {team_b_label}: {j}" for j in range(matrix_size)]
            )

            # Renderizado de la matriz de calor nativa en Streamlit con gradiente de color
            st.dataframe(
                df_matrix.style.background_gradient(cmap='YlGn', axis=None).format("{:.1f}%"),
                use_container_width=True
            )
    else:
        st.info(
            "El archivo 'direct_knockout_goal_probabilities.csv' no está disponible para desplegar los detalles de probabilidad por gol.")

# --- TAB 4: EFECTIVIDAD Y SEGUIMIENTO ---
with tab4:
    st.subheader("🎯 Rendimiento y Efectividad del Modelo")


    # Definir estructura de mapeo para etapas de eliminación directa
    STAGE_MAP = {
        "M73": "1/16 Final", "M74": "1/16 Final", "M75": "1/16 Final", "M76": "1/16 Final",
        "M77": "1/16 Final", "M78": "1/16 Final", "M79": "1/16 Final", "M80": "1/16 Final",
        "M81": "1/16 Final", "M82": "1/16 Final", "M83": "1/16 Final", "M84": "1/16 Final",
        "M85": "1/16 Final", "M86": "1/16 Final", "M87": "1/16 Final", "M88": "1/16 Final",
        "M89": "Octavos de Final", "M90": "Octavos de Final", "M91": "Octavos de Final", "M92": "Octavos de Final",
        "M93": "Octavos de Final", "M94": "Octavos de Final", "M95": "Octavos de Final", "M96": "Octavos de Final",
        "M97": "Cuartos de Final", "M98": "Cuartos de Final", "M99": "Cuartos de Final", "M100": "Cuartos de Final",
        "M101": "Semifinal", "M102": "Semifinal",
        "M103": "Tercer Puesto",
        "M104": "Final"
    }

    # Intentar detectar dinámicamente columnas de probabilidad en fase de grupos (Win, Draw, Lose)
    prob_cols = {'A': None, 'Draw': None, 'B': None}
    has_prob_cols = False

    if not df_matches_all.empty:
        for col in df_matches_all.columns:
            cl = col.lower()
            if any(kw in cl for kw in
                   ['team_a_win_pct', 'win_a_pct', 'prob_a', 'win_prob_a', 'home_win', 'team_a_win', 'win_a']):
                prob_cols['A'] = col
            elif any(kw in cl for kw in ['draw_pct', 'prob_draw', 'draw_prob', 'empat', 'draw']):
                prob_cols['Draw'] = col
            elif any(kw in cl for kw in
                     ['team_b_win_pct', 'win_b_pct', 'prob_b', 'win_prob_b', 'away_win', 'team_b_win', 'win_b',
                      'loss_prob']):
                prob_cols['B'] = col

        if prob_cols['A'] and prob_cols['Draw'] and prob_cols['B']:
            has_prob_cols = True


    # 1. PROCESAR FASE DE GRUPOS
    group_evaluated = 0
    group_winner_hits = 0
    group_perfect_scores = 0
    group_details = []

    if not df_matches_all.empty:
        for _, row in df_matches_all.iterrows():
            ta_slug = row['Team_A']
            tb_slug = row['Team_B']
            proj_a = int(row['Goals_A'])
            proj_b = int(row['Goals_B'])

            real_score = find_real_score_exact(ta_slug, tb_slug, df_grupos_resultados)
            if real_score is not None:
                real_a, real_b = real_score
                group_evaluated += 1

                # Determinar el resultado real (1, X, 2)
                real_diff = real_a - real_b
                if real_diff > 0:
                    real_outcome = "1"
                elif real_diff < 0:
                    real_outcome = "2"
                else:
                    real_outcome = "X"

                # Determinar el resultado proyectado por probabilidades o fallback por marcador
                if has_prob_cols:
                    try:
                        p_a = float(row[prob_cols['A']])
                        p_draw = float(row[prob_cols['Draw']])
                        p_b = float(row[prob_cols['B']])

                        if p_a > p_draw and p_a > p_b:
                            proj_outcome = "1"
                        elif p_b > p_a and p_b > p_draw:
                            proj_outcome = "2"
                        else:
                            proj_outcome = "X"
                    except Exception:
                        proj_diff = proj_a - proj_b
                        proj_outcome = "1" if proj_diff > 0 else ("2" if proj_diff < 0 else "X")
                else:
                    proj_diff = proj_a - proj_b
                    proj_outcome = "1" if proj_diff > 0 else ("2" if proj_diff < 0 else "X")

                is_outcome_correct = (proj_outcome == real_outcome)
                is_exact_correct = (proj_a == real_a) and (proj_b == real_b)

                if is_outcome_correct:
                    group_winner_hits += 1
                if is_exact_correct:
                    group_perfect_scores += 1

                label_map = {"1": "Victoria Local", "X": "Empate", "2": "Victoria Visitante"}

                group_details.append({
                    "Partido": f"{get_display_name(ta_slug)} vs {get_display_name(tb_slug)}",
                    "Grupo": f"Grupo {row['Group']}",
                    "Marcador Proyectado": f"{proj_a} - {proj_b}",
                    "Signo Proyectado": label_map[proj_outcome],
                    "Resultado Real": f"{real_a} - {real_b}",
                    "Acierto Ganador/Signo": "✅ Sí" if is_outcome_correct else "❌ No",
                    "Marcador Perfecto": "🎯 ¡Sí!" if is_exact_correct else "❌ No"
                })

    # 2. PROCESAR ELIMINACIÓN DIRECTA (KNOCKOUT)
    ko_evaluated = 0
    ko_winner_hits = 0
    ko_perfect_scores = 0
    ko_details = []

    if not df_real_ko.empty and bracket_data:
        for _, row in df_real_ko.iterrows():
            m_id = str(row.get('Match_ID', '')).strip()
            if m_id not in bracket_data:
                continue

            m_proj = bracket_data[m_id]
            proj_winner = str(m_proj.get('predicted_winner', '')).strip().lower()
            proj_score = m_proj.get('predicted_score', '?-?')
            matchup = m_proj.get('matchup', 'TBD vs TBD')

            # Obtener nombres proyectados
            teams_proj = matchup.split(' vs ')
            t_a_proj_slug = teams_proj[0].strip().lower() if len(teams_proj) > 0 else ""
            t_b_proj_slug = teams_proj[1].strip().lower() if len(teams_proj) > 1 else ""

            # Obtener reales del DataFrame
            t_a_real_slug = str(row.get('Team_A', '')).strip().lower()
            t_b_real_slug = str(row.get('Team_B', '')).strip().lower()

            real_goals_a = row.get('Goals_A', None)
            real_goals_b = row.get('Goals_B', None)
            real_winner = str(row.get('Winner', '')).strip().lower()

            # Evitar partidos que todavía no se juegan o no tienen datos numéricos
            if pd.isna(real_goals_a) or pd.isna(real_goals_b) or str(real_goals_a).strip() == "" or str(
                    real_goals_b).strip() == "":
                continue

            try:
                real_goals_a = int(float(real_goals_a))
                real_goals_b = int(float(real_goals_b))
            except ValueError:
                continue

            ko_evaluated += 1

            # Parsear proyección
            nums = re.findall(r'\d+', proj_score)
            pred_goals_a = int(nums[0]) if len(nums) > 0 else 0
            pred_goals_b = int(nums[1]) if len(nums) > 1 else 0

            # Alinear los goles para la comparación si el orden de los equipos se invirtió
            aligned_pred_goals_a = pred_goals_a
            aligned_pred_goals_b = pred_goals_b
            if t_a_real_slug == t_b_proj_slug and t_b_real_slug == t_a_proj_slug:
                aligned_pred_goals_a = pred_goals_b
                aligned_pred_goals_b = pred_goals_a

            is_exact_correct = (aligned_pred_goals_a == real_goals_a) and (aligned_pred_goals_b == real_goals_b)
            is_winner_correct = (proj_winner == real_winner)

            if is_winner_correct:
                ko_winner_hits += 1
            if is_exact_correct:
                ko_perfect_scores += 1

            stage_name = STAGE_MAP.get(m_id, "Eliminatoria")

            ko_details.append({
                "Partido ID": m_id,
                "Etapa": stage_name,
                "Partido": f"{get_display_name(t_a_real_slug)} vs {get_display_name(t_b_real_slug)}",
                "Proyección": f"{pred_goals_a} - {pred_goals_b}",
                "Resultado Real": f"{real_goals_a} - {real_goals_b}",
                "Ganador Proyectado": get_display_name(proj_winner),
                "Ganador Real": get_display_name(real_winner),
                "Acierto Clasificado": "✅ Sí" if is_winner_correct else "❌ No",
                "Marcador Perfecto": "🎯 ¡Sí!" if is_exact_correct else "❌ No"
            })

    # 3. METRICAS TOTALES (GLOBALES)
    total_evaluated = group_evaluated + ko_evaluated
    total_winner_hits = group_winner_hits + ko_winner_hits
    total_perfect_scores = group_perfect_scores + ko_perfect_scores

    pct_winner_hits = (total_winner_hits / total_evaluated * 100) if total_evaluated > 0 else 0.0
    pct_perfect_scores = (total_perfect_scores / total_evaluated * 100) if total_evaluated > 0 else 0.0

    # Layout de resumen general
    st.markdown("### Resumen Global de Desempeño")
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    with c_m1:
        st.metric("Partidos Disputados", value=total_evaluated)
    with c_m2:
        st.metric("Aciertos de Ganador/Signo", value=f"{total_winner_hits}", delta=f"{pct_winner_hits:.1f}% de acierto")
    with c_m3:
        st.metric("Fallos Registrados", value=f"{total_evaluated - total_winner_hits}")
    with c_m4:
        st.metric("Marcadores Perfectos", value=f"{total_perfect_scores}",
                  delta=f"{pct_perfect_scores:.1f}% de acierto")

    st.write("---")

    # Separar visualización por fases
    sub_tab_g, sub_tab_ko = st.tabs(["📋 Detalles Fase de Grupos", "🏆 Detalles Eliminatorias"])

    with sub_tab_g:
        if group_evaluated > 0:
            st.markdown(f"**Partidos evaluados en fase de grupos:** {group_evaluated}")

            df_g_details = pd.DataFrame(group_details)
            st.dataframe(
                df_g_details,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Aún no se registran resultados completados en la fase de grupos para evaluar.")

    with sub_tab_ko:
        if ko_evaluated > 0:
            st.markdown(f"**Partidos evaluados en fase de eliminación:** {ko_evaluated}")
            st.markdown(
                f"**Aciertos de clasificado:** {ko_winner_hits} | **Marcadores Perfectos:** {ko_perfect_scores}")

            df_ko_details = pd.DataFrame(ko_details)
            st.dataframe(
                df_ko_details,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Aún no se registran resultados procesados en el bracket de eliminación directa.")