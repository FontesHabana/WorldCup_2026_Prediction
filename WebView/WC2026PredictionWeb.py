import streamlit as st
import pandas as pd
import json
import os
import glob
from PIL import Image

# --- CONFIGURACIÓN Y RUTAS ---
st.set_page_config(
    page_title="Mundial 2026 - Predicciones Dinámicas",
    page_icon="🏆",
    layout="wide"
)

# Ajuste de rutas para entorno local
BASE_DIR = "/home/fontes/workspace/personal_projects/WorldCup_2026_Prediction"
RESULTS_DIR = os.path.join(BASE_DIR, "simulation_results")

PATH_FASES = os.path.join(RESULTS_DIR, "probabilidades_fases.csv")
PATH_GRUPOS = os.path.join(RESULTS_DIR, "probabilidades_grupos.csv")
PATH_BRACKET = os.path.join(RESULTS_DIR, "bracket_mas_probable.json")
PATH_CHART = os.path.join(RESULTS_DIR, "grafico_probabilidades.png")
PATH_MATCHES = os.path.join(RESULTS_DIR, "partidos_mas_probables_por_grupo.csv")

# --- ESTILOS ---
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
st.markdown("<p class='sub-titulo'>Predicciones Dinámicas Basadas en Simulación AI</p>", unsafe_allow_html=True)

# --- CARGA DINÁMICA DE METADATA ---
@st.cache_data
def get_dynamic_metadata():
    if not os.path.exists(PATH_MATCHES):
        return {}, {}
    
    df = pd.read_csv(PATH_MATCHES)
    mapping = {}
    grupos = {}
    
    for _, row in df.iterrows():
        g = f"Grupo {row['Group']}"
        t1, t2 = row['Team_A'].lower(), row['Team_B'].lower()
        
        mapping[t1] = t1.replace('_', ' ').title()
        mapping[t2] = t2.replace('_', ' ').title()
        
        if g not in grupos:
            grupos[g] = set()
        grupos[g].add(t1)
        grupos[g].add(t2)
        
    return mapping, {k: sorted(list(v)) for k, v in grupos.items()}

TEAM_MAPPING_DYNAMIC, GRUPOS_TEAMS_DYNAMIC = get_dynamic_metadata()

def get_display_name(slug):
    if not slug or pd.isna(slug): return "TBD"
    slug_clean = str(slug).lower().strip()
    return TEAM_MAPPING_DYNAMIC.get(slug_clean, slug_clean.title())

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    df_fases = pd.read_csv(PATH_FASES) if os.path.exists(PATH_FASES) else pd.DataFrame()
    if not df_fases.empty and (df_fases.columns[0] == "Unnamed: 0" or "Team" not in df_fases.columns):
        df_fases = df_fases.rename(columns={df_fases.columns[0]: "Team"})

    df_grupos = pd.read_csv(PATH_GRUPOS) if os.path.exists(PATH_GRUPOS) else pd.DataFrame()
    
    bracket = {}
    if os.path.exists(PATH_BRACKET):
        with open(PATH_BRACKET, 'r') as f:
            bracket = json.load(f)

    df_matches = pd.read_csv(PATH_MATCHES) if os.path.exists(PATH_MATCHES) else pd.DataFrame()
    
    return df_fases, df_grupos, bracket, df_matches

df_fases, df_grupos, bracket_data, df_matches_all = load_data()

# --- PESTAÑAS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Probabilidades",
    "📅 Fase de Grupos",
    "🌳 Bracket KO",
    "🔮 Validación",
    "📝 Mi Calendario"
])

# --- TAB 1: PROBABILIDADES ---
with tab1:
    st.subheader("Probabilidades de Éxito por Equipo")
    if not df_fases.empty:
        display_df = df_fases.copy()
        display_df['Team'] = display_df['Team'].apply(get_display_name)
        display_df = display_df.set_index('Team')
        st.dataframe(display_df.style.background_gradient(cmap='YlGn', axis=None).format("{:.2f}%"), 
                     height=600, use_container_width=True)
    
    if os.path.exists(PATH_CHART):
        st.image(Image.open(PATH_CHART), caption="Mapa de Probabilidades Global", use_container_width=True)

# --- TAB 2: FASE DE GRUPOS ---
with tab2:
    grupo_sel = st.selectbox("Selecciona un Grupo:", sorted(list(GRUPOS_TEAMS_DYNAMIC.keys())))
    g_id = grupo_sel.split(" ")[1]
    
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.markdown(f"### Tabla Proyectada {grupo_sel}")
        tabla_path = os.path.join(RESULTS_DIR, f"tabla_mas_probable_{grupo_sel.lower().replace(' ', '_')}.csv")
        if os.path.exists(tabla_path):
            df_t = pd.read_csv(tabla_path)
            df_t['Team'] = df_t['Team'].apply(get_display_name)
            st.table(df_t.set_index('Pos'))
        
        st.markdown("### Probabilidades de Posición")
        if not df_grupos.empty:
            equipos_g = GRUPOS_TEAMS_DYNAMIC[grupo_sel]
            df_g_prob = df_grupos[df_grupos['Team'].isin(equipos_g)].copy()
            df_g_prob['Team'] = df_g_prob['Team'].apply(get_display_name)
            st.dataframe(df_g_prob.set_index('Team').style.background_gradient(cmap='Blues', axis=1).format("{:.1f}%"))

    with c2:
        st.markdown(f"### Partidos Proyectados {grupo_sel}")
        if not df_matches_all.empty:
            df_m_g = df_matches_all[df_matches_all['Group'] == g_id]
            for _, row in df_m_g.iterrows():
                res_color = "#e8f5e9" if row['Is_Group_Most_Probable'] else "white"
                st.markdown(f"""
                <div style='border:1px solid #ddd; padding:8px; border-radius:5px; margin-bottom:5px; background-color:{res_color}'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span>{get_display_name(row['Team_A'])}</span>
                        <b>{int(row['Goals_A'])} - {int(row['Goals_B'])}</b>
                        <span>{get_display_name(row['Team_B'])}</span>
                    </div>
                    <div style='font-size:0.8rem; color:grey; text-align:center;'>Confianza: {row['Score_Probability_Pct']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 3: BRACKET ---
with tab3:
    st.subheader("Camino a la Final")
    
    def match_box(m_id):
        m = bracket_data.get(m_id)
        if not m: return st.empty()
        
        winner = m.get('predicted_winner')
        score = m.get('predicted_score', '?-?')
        # Limpieza de score si trae nombres de equipos
        if winner and winner.lower() in score.lower():
            # Intentar extraer solo números si es posible
            import re
            nums = re.findall(r'\d+', score)
            if len(nums) >= 2: score = f"{nums[0]} - {nums[1]}"
            
        st.markdown(f"""
        <div class="bracket-match">
            <div style="font-size:0.8rem; color:gray;">{m_id.replace('M', 'Partido ')}</div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9rem;">{m.get('matchup', 'TBD').replace(' vs ', '<br>').title()}</span>
                <b style="font-size:1.1rem; margin-left:10px;">{score}</b>
            </div>
            <div class="match-winner" style="margin-top:5px; border-top:1px solid #eee; padding-top:3px;">
                🏆 {get_display_name(winner)} ({m.get('winner_probability_pct', 0):.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("<div class='group-header'>1/16 Final</div>", unsafe_allow_html=True)
        for i in range(73, 81): match_box(f"M{i}")
    with col2:
        st.markdown("<div class='group-header'>1/16 Final</div>", unsafe_allow_html=True)
        for i in range(81, 89): match_box(f"M{i}")
    with col3:
        st.markdown("<div class='group-header'>Octavos</div>", unsafe_allow_html=True)
        for i in range(89, 97): match_box(f"M{i}")
    with col4:
        st.markdown("<div class='group-header'>Cuartos</div>", unsafe_allow_html=True)
        for i in range(97, 101): match_box(f"M{i}")
    with col5:
        st.markdown("<div class='group-header'>Finales</div>", unsafe_allow_html=True)
        match_box("M101")
        match_box("M102")
        st.markdown("---")
        match_box("M104")
        final_winner = bracket_data.get("M104", {}).get("predicted_winner")
        if final_winner:
            st.success(f"### CAMPEÓN: {get_display_name(final_winner)}")

# --- TAB 4: VALIDACIÓN ---
with tab4:
    st.subheader("Tu vs La Inteligencia Artificial")
    
    all_teams = sorted(list(TEAM_MAPPING_DYNAMIC.values()))
    c_user, c_ai = st.columns(2)
    
    with c_user:
        tu_campeon = st.selectbox("¿Quién será el Campeón?", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
        tu_finalista = st.selectbox("¿Quién pierde la final?", [t for t in all_teams if t != tu_campeon])
        
    with c_ai:
        ai_champ = get_display_name(bracket_data.get("M104", {}).get("predicted_winner"))
        ai_matchup = bracket_data.get("M104", {}).get("matchup", "").lower()
        ai_runner = "TBD"
        if ai_champ.lower() in ai_matchup:
            parts = ai_matchup.split(" vs ")
            for p in parts:
                if get_display_name(p) != ai_champ:
                    ai_runner = get_display_name(p)
                    break
        
        st.metric("Campeón AI", ai_champ)
        st.metric("Subcampeón AI", ai_runner)
        
    score = 0
    if tu_campeon == ai_champ: score += 50
    if tu_finalista == ai_runner: score += 50
    
    st.progress(score/100)
    st.write(f"Tu nivel de coincidencia con el modelo es del **{score}%**")

# --- TAB 5: MI CALENDARIO & ACIERTOS ---
with tab5:
    st.subheader("Predicción de Resultados Paso a Paso")
    
    if 'user_results' not in st.session_state:
        st.session_state.user_results = {}

    g_sel_cal = st.selectbox("Selecciona Grupo para rellenar:", sorted(list(GRUPOS_TEAMS_DYNAMIC.keys())), key="cal_g")
    df_g_cal = df_matches_all[df_matches_all['Group'] == g_sel_cal.split(" ")[1]]
    
    hits = 0
    perfects = 0
    total_games = 0
    
    for _, row in df_g_cal.iterrows():
        m_id = f"{row['Team_A']}_{row['Team_B']}"
        c1, c2, c3, c4, c5 = st.columns([2, 1, 0.5, 1, 2])
        
        c1.write(get_display_name(row['Team_A']))
        val_a = st.session_state.user_results.get(f"{m_id}_a", 0)
        u_a = c2.number_input("", 0, 15, val_a, key=f"in_{m_id}_a", label_visibility="collapsed")
        
        c3.write("vs")
        
        val_b = st.session_state.user_results.get(f"{m_id}_b", 0)
        u_b = c4.number_input("", 0, 15, val_b, key=f"in_{m_id}_b", label_visibility="collapsed")
        c5.write(get_display_name(row['Team_B']))
        
        st.session_state.user_results[f"{m_id}_a"] = u_a
        st.session_state.user_results[f"{m_id}_b"] = u_b
        
        # Validación vs AI
        ai_a, ai_b = int(row['Goals_A']), int(row['Goals_B'])
        
        user_win = "A" if u_a > u_b else ("B" if u_b > u_a else "D")
        ai_win = "A" if ai_a > ai_b else ("B" if ai_b > ai_a else "D")
        
        status_text = ""
        if u_a == ai_a and u_b == ai_b:
            status_text = "🎯 **¡MARCADOR EXACTO!**"
            perfects += 1
            hits += 1
        elif user_win == ai_win:
            status_text = "✅ Ganador acertado"
            hits += 1
        
        if status_text:
            st.caption(status_text)
        total_games += 1
        st.divider()

    st.sidebar.header("🏆 Tu Puntuación")
    st.sidebar.metric("Aciertos (Ganador)", f"{hits}/{total_games}")
    st.sidebar.metric("Marcadores Exactos", f"{perfects}")
    if total_games > 0:
        st.sidebar.progress(hits/total_games)
