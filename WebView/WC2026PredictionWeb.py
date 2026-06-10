import streamlit as st
import pandas as pd

# 1. Configuración de la página
st.set_page_config(
    page_title="Mundial 2026 - Control de Predicciones y Resultados",
    page_icon="🏆",
    layout="wide"
)

# Estilos personalizados sencillos
st.markdown("""
<style>
    .titulo-principal {
        color: #1b5e20;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-titulo {
        color: #555555;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .card-metrica {
        background-color: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_html=True)

st.markdown("<h1 class='titulo-principal'>🏆 Copa Mundial de la FIFA 2026</h1>", unsafe_html=True)
st.markdown(
    "<p class='sub-titulo'>Análisis de Resultados Oficiales, Predicciones y Probabilidades de Clasificación</p>",
    unsafe_html=True)

# 2. Definición de los 12 Grupos Oficiales de 4 Equipos
GRUPOS_TEAMS = {
    "Grupo A": ["México", "Sudáfrica", "Corea del Sur", "Dinamarca"],
    "Grupo B": ["Canadá", "Suiza", "Qatar", "Italia"],
    "Grupo C": ["Brasil", "Marruecos", "Haití", "Escocia"],
    "Grupo D": ["EE. UU.", "Paraguay", "Australia", "Turquía"],
    "Grupo E": ["Alemania", "Curazao", "Costa de Marfil", "Ecuador"],
    "Grupo F": ["Países Bajos", "Japón", "Ucrania", "Túnez"],
    "Grupo G": ["Bélgica", "Egipto", "Irán", "Nueva Zelanda"],
    "Grupo H": ["España", "Cabo Verde", "Arabia Saudita", "Uruguay"],
    "Grupo I": ["Francia", "Senegal", "Irak", "Noruega"],
    "Grupo J": ["Argentina", "Argelia", "Austria", "Jordania"],
    "Grupo K": ["Portugal", "Colombia", "Uzbekistán", "Rep. Dem. Congo"],
    "Grupo L": ["Inglaterra", "Croacia", "Ghana", "Panamá"]
}


# 3. Generación dinámica de la Fase de Grupos (6 partidos por grupo)
@st.cache_data
def generar_partidos_base():
    partidos = []
    id_partido = 1

    # Algunos resultados reales iniciales de prueba (ejemplo de la Jornada 1)
    resultados_reales_mock = {
        ("México", "Sudáfrica"): (2, 1),
        ("Canadá", "Suiza"): (0, 2),
        ("Brasil", "Marruecos"): (2, 2),
        ("EE. UU.", "Paraguay"): (1, 1),
        ("Alemania", "Curazao"): (3, 0),
        ("Países Bajos", "Japón"): (2, 1),
        ("Bélgica", "Egipto"): (2, 0),
        ("España", "Cabo Verde"): (3, 0),
        ("Francia", "Senegal"): (2, 0),
        ("Argentina", "Argelia"): (2, 1),
        ("Portugal", "Colombia"): (3, 1),
        ("Inglaterra", "Croacia"): (2, 0),
    }

    # Predicciones iniciales por defecto cargadas en el sistema para ilustrar el funcionamiento
    predicciones_mock = {
        ("México", "Sudáfrica"): (2, 0),  # Acierto de ganador, marcador incorrecto
        ("Canadá", "Suiza"): (1, 2),  # Acierto de ganador, marcador incorrecto
        ("Brasil", "Marruecos"): (3, 1),  # Fallo total (se pronosticó victoria y fue empate)
        ("EE. UU.", "Paraguay"): (1, 1),  # Acierto Exacto
        ("Alemania", "Curazao"): (4, 0),  # Acierto de ganador
        ("Países Bajos", "Japón"): (2, 1),  # Acierto Exacto
        ("Bélgica", "Egipto"): (1, 1),  # Fallo
        ("España", "Cabo Verde"): (3, 0),  # Acierto Exacto
        ("Francia", "Senegal"): (2, 1),  # Acierto de ganador
        ("Argentina", "Argelia"): (2, 1),  # Acierto Exacto
        ("Portugal", "Colombia"): (2, 2),  # Fallo
        ("Inglaterra", "Croacia"): (1, 0),  # Acierto de ganador
    }

    for grupo, equipos in GRUPOS_TEAMS.items():
        t1, t2, t3, t4 = equipos
        # Enfrentamientos round-robin estándar para 4 equipos
        rondas = [
            (t1, t2), (t3, t4),
            (t1, t3), (t4, t2),
            (t4, t1), (t2, t3)
        ]
        for loc, vis in rondas:
            g_l_of = resultados_reales_mock.get((loc, vis), None)
            g_v_of = resultados_reales_mock.get((loc, vis), None)
            if g_l_of is not None:
                g_l_of, g_v_of = resultados_reales_mock[(loc, vis)]

            g_l_pr, g_v_pr = predicciones_mock.get((loc, vis), (None, None))

            partidos.append({
                "id": id_partido,
                "grupo": grupo,
                "local": loc,
                "visitante": vis,
                "goles_l_oficial": g_l_of,
                "goles_v_oficial": g_v_of,
                "goles_l_pred": g_l_pr,
                "goles_v_pred": g_v_pr
            })
            id_partido += 1
    return partidos


# Inicializar partidos en Session State para mantener persistencia
if 'partidos' not in st.session_state:
    st.session_state.partidos = generar_partidos_base()

# Sincronización de los inputs de predicción ingresados por el usuario
for p in st.session_state.partidos:
    key_l = f"l_p_{p['id']}"
    key_v = f"v_p_{p['id']}"
    if key_l in st.session_state:
        p["goles_l_pred"] = st.session_state[key_l]
    if key_v in st.session_state:
        p["goles_v_pred"] = st.session_state[key_v]


# 4. Función para calcular las tablas de posiciones (oficial o proyectada)
def calcular_tabla_grupo(grupo, partidos, modo="oficial"):
    teams = GRUPOS_TEAMS[grupo]
    tabla = {t: {"PJ": 0, "G": 0, "E": 0, "P": 0, "GF": 0, "GC": 0, "DG": 0, "Pts": 0} for t in teams}

    for p in partidos:
        if p["grupo"] != grupo:
            continue

        if modo == "oficial":
            g_l, g_v = p["goles_l_oficial"], p["goles_v_oficial"]
        else:
            # En el modo predicción, si el usuario la definió se usa, sino se recurre al oficial como base
            g_l = p["goles_l_pred"] if p["goles_l_pred"] is not None else p["goles_l_oficial"]
            g_v = p["goles_v_pred"] if p["goles_v_pred"] is not None else p["goles_v_oficial"]

        if g_l is None or g_v is None:
            continue

        tabla[p["local"]]["PJ"] += 1
        tabla[p["visitante"]]["PJ"] += 1
        tabla[p["local"]]["GF"] += g_l
        tabla[p["local"]]["GC"] += g_v
        tabla[p["visitante"]]["GF"] += g_v
        tabla[p["visitante"]]["GC"] += g_l

        if g_l > g_v:
            tabla[p["local"]]["G"] += 1
            tabla[p["local"]]["Pts"] += 3
            tabla[p["visitante"]]["P"] += 1
        elif g_l == g_v:
            tabla[p["local"]]["E"] += 1
            tabla[p["local"]]["Pts"] += 1
            tabla[p["visitante"]]["E"] += 1
            tabla[p["visitante"]]["Pts"] += 1
        else:
            tabla[p["visitante"]]["G"] += 1
            tabla[p["visitante"]]["Pts"] += 3
            tabla[p["local"]]["P"] += 1

    for t in tabla:
        tabla[t]["DG"] = tabla[t]["GF"] - tabla[t]["GC"]

    df = pd.DataFrame.from_dict(tabla, orient="index")
    df = df.sort_values(by=["Pts", "DG", "GF"], ascending=False)
    df.index.name = "Equipo"
    return df.reset_index()


# 5. Diseño de Navegación por Pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard y Aciertos",
    "📅 Registro de Predicciones",
    "🔑 Clasificación de Grupos",
    "🔮 Favoritos y Campeón Proyectado"
])

# ================= TAB 1: DASHBOARD Y ACIERTOS =================
with tab1:
    st.subheader("Análisis de Rendimiento de Predicciones")

    # Filtrar partidos que tienen tanto resultado oficial como predicción del usuario
    partidos_evaluados = []
    goles_reales_totales = 0
    goles_predichos_totales = 0
    aciertos_exactos = 0
    aciertos_ganador = 0  # Incluye empate acertado

    for p in st.session_state.partidos:
        if p["goles_l_oficial"] is not None and p["goles_l_pred"] is not None:
            partidos_evaluados.append(p)
            goles_reales_totales += p["goles_l_oficial"] + p["goles_v_oficial"]
            goles_predichos_totales += p["goles_l_pred"] + p["goles_v_pred"]

            # Resultado real vs predicho
            sig_real = 1 if p["goles_l_oficial"] > p["goles_v_oficial"] else (
                -1 if p["goles_l_oficial"] < p["goles_v_oficial"] else 0)
            sig_pred = 1 if p["goles_l_pred"] > p["goles_v_pred"] else (
                -1 if p["goles_l_pred"] < p["goles_v_pred"] else 0)

            if sig_real == sig_pred:
                aciertos_ganador += 1
            if p["goles_l_oficial"] == p["goles_l_pred"] and p["goles_v_oficial"] == p["goles_v_pred"]:
                aciertos_exactos += 1

    total_eval = len(partidos_evaluados)

    if total_eval > 0:
        pct_ganador = (aciertos_ganador / total_eval) * 100
        pct_exacto = (aciertos_exactos / total_eval) * 100

        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Partidos Evaluados", f"{total_eval}")
        with col2:
            st.metric("Acierto de Ganador/Empate", f"{pct_ganador:.1f}%", f"{aciertos_ganador} correctos")
        with col3:
            st.metric("Marcadores Exactos", f"{aciertos_exactos}", f"{pct_exacto:.1f}% efectividad")
        with col4:
            st.metric("Goles Totales (Real vs Pred)", f"{goles_reales_totales}",
                      f"Predichos: {goles_predichos_totales}")

        # Gráfico comparativo de goles
        st.write("### Comparativa de Goles por Partido Evaluado")
        chart_data = []
        for p in partidos_evaluados:
            chart_data.append({
                "Partido": f"{p['local']} vs {p['visitante']}",
                "Goles Reales": p["goles_l_oficial"] + p["goles_v_oficial"],
                "Goles Predichos": p["goles_l_pred"] + p["goles_v_pred"]
            })
        df_chart = pd.DataFrame(chart_data).set_index("Partido")
        st.bar_chart(df_chart)

        # Listado detallado de aciertos y fallos
        st.write("### Desglose de Resultados Evaluados")
        detalles = []
        for p in partidos_evaluados:
            # Evaluar estado
            es_exacto = p["goles_l_oficial"] == p["goles_l_pred"] and p["goles_v_oficial"] == p["goles_v_pred"]
            sig_real = 1 if p["goles_l_oficial"] > p["goles_v_oficial"] else (
                -1 if p["goles_l_oficial"] < p["goles_v_oficial"] else 0)
            sig_pred = 1 if p["goles_l_pred"] > p["goles_v_pred"] else (
                -1 if p["goles_l_pred"] < p["goles_v_pred"] else 0)
            es_ganador = sig_real == sig_pred

            estado = "✅ Exacto" if es_exacto else ("🟢 Ganador/Empate" if es_ganador else "❌ Fallado")

            detalles.append({
                "Grupo": p["grupo"],
                "Partido": f"{p['local']} vs {p['visitante']}",
                "Resultado Oficial": f"{p['goles_l_oficial']} - {p['goles_v_oficial']}",
                "Tu Predicción": f"{p['goles_l_pred']} - {p['goles_v_pred']}",
                "Estado": estado
            })
        st.table(pd.DataFrame(detalles))
    else:
        st.info(
            "No hay partidos evaluados en este momento. Ingresa resultados oficiales y predicciones en la siguiente pestaña.")

# ================= TAB 2: REGISTRO DE PREDICCIONES =================
with tab2:
    st.subheader("Ingresa y edita tus predicciones")
    st.write(
        "Selecciona un grupo para registrar tus pronósticos. Los cambios se guardan automáticamente en tu sesión actual.")

    grupo_sel = st.selectbox("Selecciona el Grupo:", list(GRUPOS_TEAMS.keys()))
    partidos_filtrados = [p for p in st.session_state.partidos if p["grupo"] == grupo_sel]

    st.write(f"#### Partidos del {grupo_sel}")

    # Formulario dinámico para evitar sobrecargar los reruns de Streamlit
    for p in partidos_filtrados:
        col_local, col_g_l, col_vs, col_g_v, col_visitante, col_oficial = st.columns([3, 1, 0.5, 1, 3, 2])

        with col_local:
            st.markdown(f"<div style='text-align: right; font-weight: bold;'>{p['local']}</div>", unsafe_html=True)

        with col_g_l:
            st.number_input(
                "",
                min_value=0,
                max_value=15,
                value=int(p["goles_l_pred"]) if p["goles_l_pred"] is not None else 0,
                key=f"l_p_{p['id']}",
                step=1,
                label_visibility="collapsed"
            )

        with col_vs:
            st.markdown("<div style='text-align: center;'>vs</div>", unsafe_html=True)

        with col_g_v:
            st.number_input(
                "",
                min_value=0,
                max_value=15,
                value=int(p["goles_v_pred"]) if p["goles_v_pred"] is not None else 0,
                key=f"v_p_{p['id']}",
                step=1,
                label_visibility="collapsed"
            )

        with col_visitante:
            st.markdown(f"<div style='text-align: left; font-weight: bold;'>{p['visitante']}</div>", unsafe_html=True)

        with col_oficial:
            if p["goles_l_oficial"] is not None:
                st.success(f"Oficial: {p['goles_l_oficial']} - {p['goles_v_oficial']}")
            else:
                st.info("Próximamente")
        st.markdown("---")

# ================= TAB 3: CLASIFICACIÓN DE GRUPOS =================
with tab3:
    st.subheader("Llaves de los Grupos y Posiciones Proyectadas")
    st.write("Compara la tabla de posiciones oficial frente a la proyectada según tus predicciones.")

    grupo_tabla_sel = st.selectbox("Selecciona un grupo para visualizar la tabla:", list(GRUPOS_TEAMS.keys()),
                                   key="grupo_tabla")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.write("#### 📊 Tabla Oficial")
        tabla_oficial = calcular_tabla_grupo(grupo_tabla_sel, st.session_state.partidos, modo="oficial")
        st.dataframe(tabla_oficial, hide_index=True, use_container_width=True)
        st.caption("Nota: Solo considera partidos con resultados cargados oficialmente.")

    with col_t2:
        st.write("#### 🔮 Tabla Proyectada (Con tus predicciones)")
        tabla_pred = calcular_tabla_grupo(grupo_tabla_sel, st.session_state.partidos, modo="prediccion")
        st.dataframe(tabla_pred, hide_index=True, use_container_width=True)
        st.caption("Nota: Combina resultados oficiales con tus predicciones guardadas.")

# ================= TAB 4: FAVORITOS Y CAMPEÓN PROYECTADO =================
with tab4:
    st.subheader("Modelos de Probabilidad y Predicción General")

    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.markdown("""
        ### 🤖 Simulación de los Modelos de Referencia
        Antes del inicio del torneo, modelos matemáticos avanzados (como la supercomputadora de **Opta Analyst** y el modelo estadístico de **Goldman Sachs**) simularon el torneo miles de veces, arrojando las siguientes probabilidades de título:
        """)

        # Datos basados en el reporte real del modelo de junio de 2026
        prob_data = {
            "Selección": ["España", "Francia", "Inglaterra", "Argentina", "Portugal", "Brasil", "Alemania"],
            "Probabilidad Opta (AI)": ["16.1%", "13.0%", "11.2%", "10.4%", "7.0%", "6.6%", "5.1%"],
            "Probabilidad Goldman Sachs": ["26.0%", "19.0%", "5.0%", "14.0%", "S/D", "8.0%", "S/D"]
        }
        st.table(pd.DataFrame(prob_data))
        st.caption("S/D: Sin Datos públicos específicos en el informe de Goldman Sachs.")

    with col_der:
        st.markdown("""
        ### 🗺️ El Torneo Más Probable (Llaves Proyectadas)
        De acuerdo con el modelo de simulación de Goldman Sachs, el camino de eliminatorias directas más probable se estructura de la siguiente manera:
        """)

        st.info("""
        * **Cuartos de Final Proyectados:**
          * España vs Turquía
          * Francia vs Colombia
          * Argentina vs EE. UU.
          * Brasil vs Inglaterra
        * **Semifinales:**
          * España vs Francia (Ganador: España)
          * Argentina vs Brasil (Ganador: Argentina)
        * **Gran Final:**
          * España vs Argentina
        * **Campeón Proyectado:** 🏆 España
        """)

    st.markdown("---")
    st.write("### 🔮 Tu Predicción Personal del Campeón")

    # Permitir al usuario elegir su propia predicción de campeón de entre todos los participantes
    todos_equipos = sorted(list(set([team for teams in GRUPOS_TEAMS.values() for team in teams])))

    col_campeon_1, col_campeon_2 = st.columns(2)
    with col_campeon_1:
        mi_campeon = st.selectbox("Selecciona tu selección favorita a Campeón:", todos_equipos,
                                  index=todos_equipos.index("Argentina"))
        nivel_confianza = st.slider("Tu nivel de confianza en esta predicción (%):", min_value=0, max_value=100,
                                    value=75)

    with col_campeon_2:
        st.write("#### Resumen de tu Pronóstico:")
        st.success(f"🏆 Has seleccionado a **{mi_campeon}** para coronarse Campeón del Mundo.")
        st.write(f"Tu grado de seguridad es del **{nivel_confianza}%**.")
        if nivel_confianza > 80:
            st.caption("¡Tienes una confianza muy alta en esta selección!")
        elif nivel_confianza < 40:
            st.caption("Tu predicción es reservada. Sabes que el Mundial de 48 equipos depara muchas sorpresas.")