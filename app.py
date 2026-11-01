# =============================================================================
# ReúsoRO · Herramienta analítica (artefacto PIA1)
# Aprovechamiento del agua de rechazo de ósmosis inversa en servicios auxiliares
# Planta industrial de envases de vidrio, Callao
#
# M1 Datos · M2 Aptitud · M3 Tanque · M4 Economía · M5 Monte Carlo · M6 Interfaz
# Las ecuaciones son las del cuaderno del Diseño de Solución (Capítulo IV).
# Ejecutar:  streamlit run app.py
# =============================================================================
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='ReúsoRO · Herramienta analítica', page_icon='💧',
                   layout='wide', initial_sidebar_state='expanded')

# ------------------------------------------------------------------ ESTILO
BG, CARD, LINE = '#080b11', '#0f1520', '#1c2433'
MINT, RED, AMBER, BLUE, TXT, MUT = '#2de3a7', '#ff5c6c', '#f5a524', '#4da3ff', '#e6edf3', '#8b97a8'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, p, label, span, div {{ font-family: 'Space Grotesk', sans-serif; }}
.stApp {{ background: radial-gradient(1200px 500px at 10% -10%, #10261f 0%, {BG} 55%); }}
section[data-testid="stSidebar"] {{ background: #0b1018; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h3 {{ font-size: .78rem; letter-spacing: .14em; color: {MINT}; text-transform: uppercase; margin-top: .6rem; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1400px; }}
.eyebrow {{ font-family: 'JetBrains Mono', monospace; font-size: .74rem; letter-spacing: .18em; color: {MINT}; }}
.h1 {{ font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: .2rem 0 .3rem; color: {TXT}; }}
.h1 span {{ color: {MINT}; }}
.sub {{ color: {MUT}; font-size: 1rem; margin-bottom: .7rem; }}
.badges span {{ display: inline-block; border: 1px solid {LINE}; background: {CARD}; color: {TXT};
  border-radius: 999px; padding: .22rem .75rem; margin: 0 .4rem .4rem 0; font-size: .78rem; }}
.badges b {{ color: {MINT}; font-family: 'JetBrains Mono', monospace; }}
.verdict {{ border-radius: 14px; padding: 1rem 1.3rem; margin: .8rem 0 1rem; display: flex; gap: 1rem; align-items: center; }}
.verdict .big {{ font-size: 1.5rem; font-weight: 700; letter-spacing: .02em; white-space: nowrap; }}
.verdict .txt {{ color: {TXT}; font-size: .95rem; }}
.v-ok  {{ background: rgba(45,227,167,.10); border: 1px solid rgba(45,227,167,.55); }}
.v-ok .big {{ color: {MINT}; }}
.v-no  {{ background: rgba(255,92,108,.10); border: 1px solid rgba(255,92,108,.6); }}
.v-no .big {{ color: {RED}; }}
.kpis {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: .8rem; margin-bottom: 1rem; }}
.kpi {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: .85rem 1rem; }}
.kpi .l {{ color: {MUT}; font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; }}
.kpi .v {{ font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; font-weight: 600; color: {TXT}; margin-top: .2rem; }}
.kpi .d {{ font-size: .78rem; color: {MUT}; margin-top: .1rem; }}
.kpi.accent {{ border-color: rgba(45,227,167,.5); }}
.kpi.accent .v {{ color: {MINT}; }}
.kpi.bad .v {{ color: {RED}; }}
.card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .8rem; }}
.card h4 {{ margin: 0 0 .35rem; font-size: 1rem; color: {TXT}; }}
.card p {{ margin: 0; color: {MUT}; font-size: .88rem; }}
.cp {{ display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: .74rem; border-radius: 6px; padding: .18rem .55rem; margin: 0 .35rem .35rem 0; }}
.cp.ok {{ background: rgba(45,227,167,.12); color: {MINT}; border: 1px solid rgba(45,227,167,.4); }}
.cp.no {{ background: rgba(255,92,108,.12); color: {RED}; border: 1px solid rgba(255,92,108,.45); }}
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ background: {CARD}; border: 1px solid {LINE}; border-bottom: none; border-radius: 10px 10px 0 0; padding: .45rem 1rem; }}
.stTabs [aria-selected="true"] {{ background: rgba(45,227,167,.12) !important; color: {MINT} !important; }}
.foot {{ color: {MUT}; font-size: .78rem; border-top: 1px solid {LINE}; padding-top: .7rem; margin-top: 1.2rem; }}
@media (max-width: 1100px) {{ .kpis {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>
""", unsafe_allow_html=True)


def fig_style(fig, h=340, legend=True):
    fig.update_layout(height=h, paper_bgcolor=CARD, plot_bgcolor=CARD, margin=dict(l=50, r=20, t=50, b=70), title_font=dict(size=14),
                      font=dict(family='Space Grotesk, sans-serif', color=TXT, size=12),
                      legend=dict(orientation='h', y=-0.32, x=0, bgcolor='rgba(0,0,0,0)') if legend else None,
                      showlegend=legend, hoverlabel=dict(bgcolor='#111a26', font_family='JetBrains Mono'))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE)
    return fig


# ------------------------------------------------------------------ MODELO
PERFIL = np.array([.02] * 6 + [.07] * 2 + [.05] * 6 + [.07] * 2 + [.04] * 6 + [.03] * 2)


def conductividad_rechazo(C_f, C_p, R):
    """Ec. 1 · balance de masa salina: C_r = (C_f − R·C_p) / (1 − R)"""
    return (C_f - R * C_p) / (1.0 - R)


def recuperacion_maxima(C_f, C_p, C_max):
    """Ec. 2 · aporte del trabajo: R_max = (C_max − C_f) / (C_max − C_p)"""
    return (C_max - C_f) / (C_max - C_p)


def caudal_reutilizable(Q_r, D, delta, apto, V_cap=np.inf):
    """Ec. 3 · regla de decisión: Q_u = min(δ·Q_r, V_cap, D) si el rechazo es apto"""
    return np.where(apto, np.minimum(np.minimum(delta * Q_r, V_cap), D), 0.0)


def ventana_tanque(Q_u, perfil, tau_max):
    """Ec. 4 · V_min por amortiguamiento del pico; V_max por tiempo de residencia"""
    acum = np.cumsum(Q_u / 24.0 - Q_u * perfil)
    return acum.max() - min(acum.min(), 0.0), Q_u * tau_max / 24.0, acum


def evaluar(I0, bruto, opex, cok, n):
    fd = 1.0 / (1.0 + cok) ** np.arange(1, n + 1)
    FAS, neto = fd.sum(), bruto - opex
    VAN = -I0 + neto * FAS
    BC = (bruto * FAS) / (I0 + opex * FAS) if (I0 + opex * FAS) else 0.0
    IR = (neto * FAS) / I0 if I0 else 0.0
    PRD = I0 / neto if neto > 0 else np.inf
    lo, hi = 1e-9, 1e4
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if -I0 + (neto / (1 + m) ** np.arange(1, n + 1)).sum() > 0 else (lo, m)
    TIR = 0.5 * (lo + hi) if neto > 0 else np.nan
    acum, prd_d = 0.0, np.nan
    for t, f in enumerate(fd, start=1):
        if neto > 0 and acum + neto * f >= I0:
            prd_d = (t - 1) + (I0 - acum) / (neto * f)
            break
        acum += neto * f
    return dict(FAS=FAS, neto=neto, VAN=VAN, TIR=TIR, PRD=PRD, PRD_d=prd_d, BC=BC, IR=IR, fd=fd)


@st.cache_data
def base_de_prueba():
    """P9 · 236 registros anclados en la media y dispersión de los registros del equipo."""
    rng = np.random.default_rng(20260926)
    N = 236
    esp = {'Q_f': ('norm', (100.70, 2.50)), 'R': ('norm', (0.7031, 0.0070)),
           'C_f': ('norm', (737.21, 42.00)), 'C_p': ('norm', (53.74, 6.50)),
           'D': ('tri', (6.60, 7.745, 8.90)), 'delta': ('tri', (0.90, 0.96, 1.00)),
           'V_cap': ('tri', (5.50, 6.10, 6.60))}
    d = {k: (rng.normal(*p, N) if t == 'norm' else rng.triangular(*p, N)) for k, (t, p) in esp.items()}
    df = pd.DataFrame(d)
    df.insert(0, 'dia', np.arange(1, N + 1))
    df['Q_p'] = df['Q_f'] * df['R']
    df['Q_r'] = df['Q_f'] * (1 - df['R'])
    return df[['dia', 'Q_f', 'Q_p', 'Q_r', 'R', 'C_f', 'C_p', 'D', 'V_cap', 'delta']].round(4)


@st.cache_data
def monte_carlo(T, apto, V_inst, cok, n, dias, n_iter=10_000, seed=20260919):
    g = np.random.default_rng(seed)
    MC = {'Q_r': g.normal(29.92, 1.04, n_iter),
          'V_cap': g.triangular(V_inst * 0.9016, V_inst, V_inst * 1.0820, n_iter),
          'D': g.triangular(6.60, 7.745, 8.90, n_iter),
          'T': g.triangular(T * 0.9575, T, T * 1.1687, n_iter),
          'I_0': g.triangular(8402.0, 9662.3, 12077.9, n_iter),
          'OPEX': g.triangular(2100, 2596, 3500, n_iter),
          'delta': g.triangular(0.90, 0.96, 1.00, n_iter)}
    Q = np.minimum.reduce([MC['delta'] * MC['Q_r'], MC['V_cap'], MC['D']]) * (1.0 if apto else 0.0)
    neto = Q * dias * MC['T'] - MC['OPEX']
    fd = 1.0 / (1.0 + cok) ** np.arange(1, n + 1)
    VAN = -MC['I_0'] + neto * fd.sum()
    rk = lambda a: pd.Series(a).rank().values
    sens = {k: float(np.corrcoef(rk(v), rk(VAN))[0, 1]) if np.std(VAN) > 0 else 0.0 for k, v in MC.items()}
    return VAN, dict(sorted(sens.items(), key=lambda kv: abs(kv[1])))


# ------------------------------------------------------------------ ENTRADAS
base = base_de_prueba()
with st.sidebar:
    st.markdown(f"<div class='eyebrow'>REÚSORO · ENTRADAS</div>", unsafe_allow_html=True)
    st.markdown('### 1 · Registros del equipo')
    arch = st.file_uploader('Cargar CSV de planta (opcional)', type='csv',
                            help='Mismas columnas que la base de prueba. Si no se carga, se usa la base de 236 registros.')
    df0 = pd.read_csv(arch) if arch is not None else base.copy()
    R_base = float(df0['R'].mean())

    st.markdown('### 2 · Operación y calidad')
    R_sel = st.slider('Recuperación de operación del equipo', 0.60, 0.85, round(R_base, 3), 0.001, format='%.3f',
                      help='Promedio de recuperación. Cada registro se desplaza para que su promedio sea este valor.')
    C_max = st.number_input('Conductividad máxima admisible (µS/cm)', 500.0, 6000.0, 3000.0, 50.0,
                            help='Límite preliminar de referencia para uso no potable.')
    st.markdown('### 3 · Instalación')
    V_inst = st.number_input('Capacidad instalada de almacenamiento y trasvase (m³/día)', 0.5, 50.0, 6.10, 0.1)
    tau_max = st.slider('Tiempo máximo de residencia (h)', 6, 72, 24)
    st.markdown('### 4 · Economía')
    T_agua = st.number_input('Tarifa de agua potable (S/ por m³)', 0.5, 30.0, 3.654, 0.001, format='%.3f')
    T_alc = st.number_input('Tarifa de alcantarillado (S/ por m³)', 0.5, 30.0, 3.448, 0.001, format='%.3f')
    esc = st.radio('Escenario de inversión', ['Equipos cedidos (S/ 3,105)', 'Precios de mercado (S/ 9,662)', 'Otro'], index=0)
    I0 = {'E': 3105.0, 'P': 9662.30}.get(esc[0], None)
    if I0 is None:
        I0 = st.number_input('Inversión inicial (S/)', 0.0, 200000.0, 5000.0, 100.0)
    OPEX = st.number_input('Gasto operativo anual (S/)', 0.0, 50000.0, 2596.32, 50.0)
    COK = st.slider('Costo de oportunidad del capital', 0.02, 0.30, 0.10, 0.01)
    N_A = st.slider('Horizonte de evaluación (años)', 1, 15, 5)
    DIAS = st.number_input('Días operativos al año', 200, 365, 360, 5)

# ------------------------------------------------------------------ PROCESO
df = df0.copy()
shift = R_sel - R_base if abs(R_sel - R_base) > 5e-4 else 0.0
df['R'] = (df['R'] + shift).clip(0.30, 0.97)
df['Q_p'] = df['Q_f'] * df['R']
df['Q_r'] = df['Q_f'] * (1 - df['R'])
T = T_agua + T_alc

# M2 · aptitud
df['C_r'] = conductividad_rechazo(df['C_f'], df['C_p'], df['R'])
df['R_max'] = recuperacion_maxima(df['C_f'], df['C_p'], C_max)
df['apto'] = df['C_r'] <= C_max
C_r_m, R_m, Rmax_m = df['C_r'].mean(), df['R'].mean(), df['R_max'].mean()
conf = df['apto'].mean()
APTO = conf >= 0.95
margen = (1 - C_r_m / C_max) * 100

# M3 · asignación y tanque
df['Q_u'] = caudal_reutilizable(df['Q_r'].values, df['D'].values, df['delta'].values, df['apto'].values)
Q_u_fis = df['Q_u'].mean()
V_min, V_max, acum = ventana_tanque(Q_u_fis, PERFIL, tau_max)
Q_u = caudal_reutilizable(df['Q_r'].values, df['D'].values, df['delta'].values,
                          df['apto'].values, V_inst).mean() if APTO else 0.0
lims = {'Rechazo disponible (δ·Qr)': (df['delta'] * df['Q_r']).mean(), 'Capacidad instalada': V_inst,
        'Demanda de servicios': df['D'].mean()}
activa = min(lims, key=lims.get)

# M4 · circularidad y economía
Q_r_m, D_m = df['Q_r'].mean(), df['D'].mean()
circ, sust = (Q_u / Q_r_m if Q_r_m else 0), min(Q_u / D_m, 1.0) if D_m else 0
bruto = Q_u * DIAS * T
ev = evaluar(I0, bruto, OPEX, COK, N_A)

# M5 · Monte Carlo
VAN_mc, SENS = monte_carlo(T, APTO, V_inst, COK, N_A, DIAS)

# ------------------------------------------------------------------ ENCABEZADO
st.markdown(f"""
<div class='eyebrow'>HERRAMIENTA ANALÍTICA · ARTEFACTO PIA1 · INGENIERÍA INDUSTRIAL – EPE · UPC</div>
<div class='h1'>Reúso<span>RO</span></div>
<div class='sub'>Aprovechamiento del agua de rechazo de ósmosis inversa en servicios auxiliares ·
planta industrial de envases de vidrio, Callao</div>
<div class='badges'>
 <span><b>{len(df)}</b> registros {'(base de prueba)' if arch is None else '(CSV cargado)'}</span>
 <span>Tarifa SEDAPAL del periodo <b>S/ {T:.3f}/m³</b></span>
 <span>Límite de calidad <b>{C_max:,.0f} µS/cm</b></span>
 <span>Escenario <b>{esc.split(' (')[0].lower()}</b></span>
</div>
""", unsafe_allow_html=True)

if APTO:
    st.markdown(f"""<div class='verdict v-ok'><div class='big'>✔ AGUA APTA PARA REÚSO</div>
    <div class='txt'>Conductividad media del rechazo <b>{C_r_m:,.0f} µS/cm</b> frente a un límite de {C_max:,.0f} µS/cm
    · margen de <b>{margen:.1f} %</b> · {conf*100:.1f} % de registros conformes.</div></div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='verdict v-no'><div class='big'>✖ NO APTO</div>
    <div class='txt'>Con una recuperación de <b>{R_m:.3f}</b> el rechazo llega a <b>{C_r_m:,.0f} µS/cm</b>, por encima del límite de
    {C_max:,.0f} µS/cm. Mantener la recuperación por debajo de <b>{Rmax_m:.3f}</b> o evaluar un tratamiento previo.</div></div>""",
                unsafe_allow_html=True)


def kpi(l, v, d, cls=''):
    return f"<div class='kpi {cls}'><div class='l'>{l}</div><div class='v'>{v}</div><div class='d'>{d}</div></div>"


prd_txt = f"{ev['PRD']*12:.1f} meses" if np.isfinite(ev['PRD']) else '—'
st.markdown("<div class='kpis'>" +
            kpi('Recuperación', f'{R_m:.3f}', f'máxima admisible {Rmax_m:.3f}', 'accent' if APTO else 'bad') +
            kpi('Reúso diario', f'{Q_u:.2f} m³', f'restricción activa: {activa.split(" (")[0].lower()}' if APTO else 'rechazo no apto') +
            kpi('Agua de red evitada', f'{sust*100:.0f} %', 'de la demanda de servicios') +
            kpi('VAN', f'S/ {ev["VAN"]:,.0f}', f'{N_A} años · COK {COK*100:.0f} %', 'accent' if ev['VAN'] > 0 else 'bad') +
            kpi('Recuperación de la inversión', prd_txt, f'TIR {ev["TIR"]*100:.1f} %' if ev['neto'] > 0 else 'sin ahorro neto') +
            "</div>", unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6 = st.tabs(['① Datos', '② Aptitud', '③ Tanque', '④ Economía', '⑤ Incertidumbre', '⑥ Reporte'])

# ------------------------------------------------------------------ M1 DATOS
with t1:
    balance_ok = (df0['Q_p'] + df0['Q_r'] - df0['Q_f']).abs().max() < 1e-3 if {'Q_p', 'Q_r'} <= set(df0) else True
    vacios = int(df0.isna().sum().sum())
    st.markdown(f"""<div class='card'><h4>M1 · Carga y validación de registros</h4>
    <span class='cp {'ok' if len(df0) > 0 else 'no'}'>CP · {len(df0)} registros cargados</span>
    <span class='cp {'ok' if vacios == 0 else 'no'}'>CP · {vacios} valores vacíos</span>
    <span class='cp {'ok' if balance_ok else 'no'}'>CP · balance Qf = Qp + Qr {'consistente' if balance_ok else 'inconsistente'}</span>
    <p style='margin-top:.4rem'>Base de prueba anclada en la media y la dispersión de los registros del equipo
    (jun. 2021 – ene. 2022). Puede reemplazarse por el CSV real de planta desde el panel izquierdo.</p></div>""",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    f = go.Figure()
    for col, name, color in [('Q_f', 'Alimentación', BLUE), ('Q_p', 'Producto', MINT), ('Q_r', 'Rechazo', RED)]:
        f.add_trace(go.Scatter(x=df['dia'], y=df[col], name=name, line=dict(color=color, width=1.6)))
    f.update_layout(title='Caudales por registro (m³/día)')
    c1.plotly_chart(fig_style(f), use_container_width=True)
    f = go.Figure()
    f.add_trace(go.Scatter(x=df['dia'], y=df['C_f'], name='Alimentación', line=dict(color=BLUE, width=1.6)))
    f.add_trace(go.Scatter(x=df['dia'], y=df['C_p'], name='Producto', line=dict(color=MINT, width=1.6)))
    f.update_layout(title='Conductividad medida (µS/cm)')
    c2.plotly_chart(fig_style(f), use_container_width=True)
    st.dataframe(df0.head(12), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ M2 APTITUD
with t2:
    c1, c2 = st.columns([1.35, 1])
    Rg = np.linspace(0.55, 0.85, 200)
    Cg = conductividad_rechazo(df['C_f'].mean(), df['C_p'].mean(), Rg)
    f = go.Figure()
    f.add_hrect(y0=0, y1=C_max, fillcolor='rgba(45,227,167,.06)', line_width=0)
    f.add_hrect(y0=C_max, y1=max(Cg.max(), C_max) * 1.05, fillcolor='rgba(255,92,108,.07)', line_width=0)
    f.add_trace(go.Scatter(x=Rg, y=Cg, name='Conductividad del rechazo', line=dict(color=BLUE, width=3)))
    f.add_hline(y=C_max, line=dict(color=RED, dash='dash'), annotation_text=f'Límite {C_max:,.0f} µS/cm',
                annotation_font_color=RED)
    f.add_vline(x=Rmax_m, line=dict(color=AMBER, dash='dot'), annotation_text=f'Rmáx {Rmax_m:.3f}',
                annotation_font_color=AMBER, annotation_position='top left')
    f.add_trace(go.Scatter(x=[R_m], y=[C_r_m], mode='markers+text', name='Operación actual',
                           marker=dict(size=16, color=MINT if APTO else RED, line=dict(color='white', width=2)),
                           text=[f'  R = {R_m:.3f}'], textposition='middle right'))
    f.update_layout(title='¿Hasta dónde puede recuperar el equipo sin que el rechazo deje de ser apto?',
                    xaxis_title='Recuperación del equipo (R)', yaxis_title='µS/cm')
    c1.plotly_chart(fig_style(f, 400), use_container_width=True)
    with c2:
        st.markdown(f"""<div class='card'><h4>M2 · Modelo matemático</h4>
        <p>Balance de sales: &nbsp;<b style='color:{TXT}'>Cr = (Cf − R·Cp) / (1 − R)</b></p>
        <p>Recuperación máxima: &nbsp;<b style='color:{MINT}'>Rmáx = (Cmáx − Cf) / (Cmáx − Cp)</b></p></div>""",
                    unsafe_allow_html=True)
        tab = pd.DataFrame({'Variable': ['Conductividad de alimentación', 'Conductividad del producto',
                                         'Conductividad del rechazo', 'Límite admisible', 'Margen de aptitud',
                                         'Recuperación de operación', 'Recuperación máxima', 'Holgura',
                                         'Registros conformes'],
                            'Valor': [f"{df['C_f'].mean():,.1f} µS/cm", f"{df['C_p'].mean():,.1f} µS/cm",
                                      f'{C_r_m:,.0f} µS/cm', f'{C_max:,.0f} µS/cm', f'{margen:.1f} %',
                                      f'{R_m:.4f}', f'{Rmax_m:.4f}', f'{Rmax_m - R_m:+.4f}', f'{conf*100:.1f} %']})
        st.dataframe(tab, hide_index=True, use_container_width=True)
        if APTO and Rmax_m - R_m < 0.02:
            st.warning('La holgura de recuperación es menor a 0.02: un ajuste del equipo podría invalidar el reúso.')
    f = go.Figure(go.Histogram(x=df['C_r'], nbinsx=30, marker_color=BLUE, opacity=.85, name='Registros'))
    f.add_vline(x=C_max, line=dict(color=RED, dash='dash'), annotation_text='Límite', annotation_font_color=RED)
    f.update_layout(title='Conductividad del rechazo en cada registro (µS/cm)', bargap=.05)
    st.plotly_chart(fig_style(f, 260, legend=False), use_container_width=True)

# ------------------------------------------------------------------ M3 TANQUE
with t3:
    c1, c2 = st.columns(2)
    names, vals = list(lims), list(lims.values())
    f = go.Figure(go.Bar(x=vals, y=names, orientation='h', text=[f'{v:.2f} m³/día' for v in vals],
                         textposition='outside',
                         marker_color=[MINT if n == activa else '#2a3546' for n in names]))
    f.update_layout(title=f'Regla de decisión: Qu = mín(δ·Qr, capacidad, demanda) → {Q_u:.2f} m³/día',
                    xaxis_range=[0, max(vals) * 1.45])
    c1.plotly_chart(fig_style(f, 300, legend=False), use_container_width=True)
    h = np.arange(24)
    f = go.Figure()
    f.add_trace(go.Bar(x=h, y=Q_u_fis * PERFIL, name='Consumo horario', marker_color=BLUE, opacity=.8))
    f.add_trace(go.Scatter(x=h, y=np.full(24, Q_u_fis / 24), name='Llegada continua al tanque',
                           line=dict(color=MINT, width=2)))
    f.add_trace(go.Scatter(x=h, y=acum, name='Volumen acumulado', line=dict(color=AMBER, dash='dot'), yaxis='y2'))
    f.update_layout(title='Perfil de tres turnos (m³/h)', xaxis_title='Hora del día',
                    yaxis2=dict(overlaying='y', side='right', showgrid=False, title='m³'))
    c2.plotly_chart(fig_style(f, 300), use_container_width=True)
    factible = V_min <= V_max
    f = go.Figure()
    f.add_trace(go.Bar(x=['Mínimo por pico', 'Instalado', 'Máximo por residencia'], y=[V_min, V_inst, V_max],
                       marker_color=[AMBER, MINT, BLUE], text=[f'{V_min:.2f}', f'{V_inst:.2f}', f'{V_max:.2f}'],
                       textposition='outside'))
    f.update_layout(title=f"Ventana factible del tanque: {V_min:.2f} – {V_max:.2f} m³ ({'factible' if factible else 'infactible'})",
                    yaxis_title='m³', yaxis_range=[0, max(V_max, V_inst) * 1.25])
    st.plotly_chart(fig_style(f, 300, legend=False), use_container_width=True)
    st.markdown(f"""<div class='card'><h4>Lectura</h4><p>La restricción activa es la <b style='color:{MINT}'>{activa.lower()}</b>.
    El tanque instalado ({V_inst:.2f} m³) cubre {V_inst / V_min if V_min else 0:.1f} veces el mínimo necesario para amortiguar el pico:
    ampliarlo no añade reúso; la vía de mejora es sumar puntos de consumo.</p></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ M4 ECONOMIA
with t4:
    c1, c2 = st.columns([1, 1.4])
    f = go.Figure(go.Bar(x=['Antes', 'Con reúso'], y=[D_m, max(D_m - Q_u, 0)], marker_color=['#5b6778', MINT],
                         text=[f'{D_m:.2f}', f'{max(D_m - Q_u, 0):.2f}'], textposition='outside'))
    f.update_layout(title=f'Agua de red en servicios auxiliares (m³/día) · −{sust*100:.0f} %', yaxis_range=[0, D_m * 1.3])
    c1.plotly_chart(fig_style(f, 330, legend=False), use_container_width=True)
    yrs = np.arange(0, N_A + 1)
    flujo = np.r_[-I0, np.full(N_A, ev['neto'])]
    desc = np.r_[-I0, ev['neto'] * ev['fd']]
    f = go.Figure()
    f.add_trace(go.Bar(x=yrs, y=flujo, name='Flujo neto', marker_color=[RED] + [MINT] * N_A))
    f.add_trace(go.Scatter(x=yrs, y=desc.cumsum(), name='Acumulado descontado', line=dict(color=AMBER, width=3)))
    f.update_layout(title='Flujo de caja (S/)', xaxis_title='Año')
    c2.plotly_chart(fig_style(f, 330), use_container_width=True)
    ced = evaluar(3105.0, bruto, OPEX, COK, N_A)
    mkt = evaluar(9662.30, bruto, OPEX, COK, N_A)
    fm = lambda e: [f"S/ {e['VAN']:,.0f}", f"{e['TIR']*100:.1f} %" if e['neto'] > 0 else '—',
                    f"{e['PRD']*12:.1f} meses" if np.isfinite(e['PRD']) else '—',
                    f"{e['PRD_d']*12:.1f} meses" if np.isfinite(e['PRD_d']) else '—', f"{e['BC']:.2f}"]
    st.dataframe(pd.DataFrame({'Indicador': ['VAN', 'TIR', 'Recuperación simple', 'Recuperación descontada', 'Beneficio/costo'],
                               'Escenario seleccionado': fm(ev), 'Equipos cedidos': fm(ced), 'Precios de mercado': fm(mkt)}),
                 hide_index=True, use_container_width=True)
    st.markdown(f"""<div class='card'><h4>Circularidad</h4><p>Tasa de circularidad <b style='color:{MINT}'>{circ*100:.1f} %</b>
    del rechazo · tasa de sustitución de agua de red <b style='color:{MINT}'>{sust*100:.1f} %</b> · ahorro bruto
    S/ {bruto:,.0f}/año · ahorro neto S/ {ev['neto']:,.0f}/año.</p></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ M5 MONTE CARLO
with t5:
    p5, p50, p95 = np.percentile(VAN_mc, [5, 50, 95])
    pos = (VAN_mc > 0).mean() * 100
    st.markdown("<div class='kpis'>" + kpi('Escenarios', '10,000', 'simulación de Monte Carlo') +
                kpi('VAN medio', f'S/ {VAN_mc.mean():,.0f}', 'precios de mercado') + kpi('P5', f'S/ {p5:,.0f}', 'escenario pesimista') +
                kpi('P95', f'S/ {p95:,.0f}', 'escenario optimista') +
                kpi('VAN > 0', f'{pos:.0f} %', 'de los escenarios', 'accent' if pos >= 95 else 'bad') + "</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.4, 1])
    f = go.Figure(go.Histogram(x=VAN_mc, nbinsx=60, marker_color=MINT, opacity=.8))
    for v, lab, col in [(p5, 'P5', AMBER), (p50, 'P50', TXT), (p95, 'P95', AMBER), (0, 'VAN = 0', RED)]:
        f.add_vline(x=v, line=dict(color=col, dash='dash'), annotation_text=lab, annotation_font_color=col)
    f.update_layout(title='Distribución del VAN (S/)', bargap=.03)
    c1.plotly_chart(fig_style(f, 360, legend=False), use_container_width=True)
    lab = {'T': 'Tarifa', 'V_cap': 'Capacidad instalada', 'OPEX': 'Gasto operativo', 'I_0': 'Inversión',
           'D': 'Demanda', 'delta': 'Disponibilidad', 'Q_r': 'Caudal de rechazo'}
    f = go.Figure(go.Bar(x=list(SENS.values()), y=[lab[k] for k in SENS], orientation='h',
                         marker_color=[MINT if v > 0 else RED for v in SENS.values()],
                         text=[f'{v:+.2f}' for v in SENS.values()], textposition='outside'))
    f.update_layout(title='¿Qué variable mueve más el VAN? (Spearman)', xaxis_range=[-1, 1])
    c2.plotly_chart(fig_style(f, 360, legend=False), use_container_width=True)
    st.caption('La simulación usa la inversión a precios de mercado, el escenario más exigente. '
               'El caudal de rechazo casi no influye en el VAN porque el reúso lo limitan la capacidad instalada y el consumo.')

# ------------------------------------------------------------------ M6 REPORTE
with t6:
    res = pd.DataFrame({'Indicador': ['Veredicto', 'Conductividad media del rechazo (µS/cm)', 'Margen de aptitud (%)',
                                      'Recuperación de operación', 'Recuperación máxima admisible', 'Reúso diario (m³/día)',
                                      'Restricción activa', 'Ventana del tanque (m³)', 'Tasa de circularidad (%)',
                                      'Agua de red evitada (%)', 'Ahorro neto anual (S/)', 'VAN (S/)', 'TIR (%)',
                                      'Recuperación de la inversión (meses)', 'Escenarios con VAN > 0 (%)'],
                        'Valor': ['APTO' if APTO else 'NO APTO', f'{C_r_m:,.0f}', f'{margen:.1f}', f'{R_m:.4f}',
                                  f'{Rmax_m:.4f}', f'{Q_u:.2f}', activa, f'{V_min:.2f} – {V_max:.2f}', f'{circ*100:.1f}',
                                  f'{sust*100:.1f}', f"{ev['neto']:,.0f}", f"{ev['VAN']:,.0f}",
                                  f"{ev['TIR']*100:.1f}" if ev['neto'] > 0 else '—', prd_txt.replace(' meses', ''),
                                  f'{pos:.1f}']})
    c1, c2 = st.columns([1.3, 1])
    c1.dataframe(res, hide_index=True, use_container_width=True, height=560)
    with c2:
        st.markdown(f"""<div class='card'><h4>Exportar resultados</h4><p>Descarga el resumen de la decisión y los
        registros con su conductividad de rechazo y su veredicto, para adjuntarlos al informe de planta.</p></div>""",
                    unsafe_allow_html=True)
        st.download_button('⬇ Resumen de la decisión (CSV)', res.to_csv(index=False).encode('utf-8-sig'),
                           'ReusoRO_resumen.csv', 'text/csv', use_container_width=True)
        st.download_button('⬇ Registros con veredicto (CSV)', df.round(4).to_csv(index=False).encode('utf-8-sig'),
                           'ReusoRO_registros.csv', 'text/csv', use_container_width=True)
        html = f"<h2>ReúsoRO · Resumen de la decisión</h2>{res.to_html(index=False)}"
        st.download_button('⬇ Reporte (HTML)', html.encode('utf-8'), 'ReusoRO_reporte.html', 'text/html',
                           use_container_width=True)

st.markdown(f"""<div class='foot'>ReúsoRO · Herramienta analítica · Ing. Industrial – EPE · UPC.
Resultados preliminares con base de prueba de 236 registros y tarifa SEDAPAL del periodo de los registros.
La conductividad del rechazo se estima por balance de masa salina y requiere verificación de laboratorio.</div>""",
            unsafe_allow_html=True)
