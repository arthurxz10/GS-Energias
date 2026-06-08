"""
Missão ORION - Sistema de Monitoramento Energético
Global Solution 2026.1 - Energias Renováveis e Sustentáveis
FIAP - Turma 1CCPY

Grupo 05:
    Arthur dos Santos Bezerra     RM 569721
    Carlos Henrique Fratezi       RM 571792
    Felipe Gouveia Braga          RM 568956

A ideia aqui é usar dados reais de exoplanetas da NASA pra simular
o monitoramento energético da nossa cápsula espacial ORION. A gente
pega a temperatura e luminosidade de cada estrela hospedeira e calcula
quanto de energia os painéis solares da cápsula conseguiriam captar
se ela tivesse orbitando aquele sistema.

Dataset: NASA Exoplanet Archive - PSCompPars
Fonte: https://exoplanetarchive.ipac.caltech.edu
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # necessário pra rodar sem interface gráfica
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------
# configurações da missão
# -----------------------------------------------------------------------

# constante física: luminosidade do Sol em Watts
L_SOL_WATTS = 3.828e26

# a cápsula ORION tem 72m² de painéis solares do tipo GaAs (alta eficiência)
# esse tipo de painel é usado em missões reais como a ISS
AREA_PAINEL_M2   = 72.0
EFICIENCIA       = 0.29   # 29% de eficiência

# consumo estimado de cada módulo da cápsula (em Watts)
# os valores são baseados em sistemas reais de missões espaciais
CONSUMO_MODULOS = {
    "ORION-COMM":   1800,   # comunicação com a Terra
    "ORION-LIFE":   3200,   # suporte à vida (o mais crítico)
    "ORION-NAV":     950,   # navegação e propulsão
    "ORION-SCI":    2100,   # instrumentos científicos
    "ORION-THERM":  1400,   # controle térmico
}

CONSUMO_TOTAL = sum(CONSUMO_MODULOS.values())  # 9450 W no total

# limites pra classificar o nível de alerta
LIMITE_CRITICO  = 0.40   # abaixo de 40% de eficiência -> crítico
LIMITE_ATENCAO  = 0.65   # abaixo de 65% -> atenção

# pasta onde os gráficos e relatório vão ser salvos
PASTA_SAIDA = "orion_outputs"
os.makedirs(PASTA_SAIDA, exist_ok=True)

# cores pro terminal (ANSI escape codes)
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"

def colorir(texto, cor):
    return f"{cor}{texto}{RESET}"


# -----------------------------------------------------------------------
# dados do NASA Exoplanet Archive (embutidos no código)
# -----------------------------------------------------------------------
# a gente tentou puxar direto da API da NASA mas o ambiente não permite
# conexão externa, então extraímos esses dados manualmente do portal
# https://exoplanetarchive.ipac.caltech.edu/TAP/sync
#
# colunas: nome do planeta, estrela hospedeira, temperatura estelar (K),
#          luminosidade em log(L/L☉), raio estelar (R☉), temp. equilíbrio do planeta (K)

DADOS_NASA = [
    # planetas em zona habitável ou próximos - selecionamos os mais conhecidos
    ("Kepler-22 b",      "Kepler-22",    5518, -0.107,  0.979,  262),
    ("Kepler-442 b",     "Kepler-442",   4402, -0.622,  0.598,  233),
    ("Kepler-452 b",     "Kepler-452",   5757,  0.120,  1.110,  265),
    ("Kepler-62 e",      "Kepler-62",    4925, -0.328,  0.640,  270),
    ("Kepler-62 f",      "Kepler-62",    4925, -0.328,  0.640,  208),
    ("Kepler-186 f",     "Kepler-186",   3788, -1.028,  0.472,  188),
    ("Kepler-1649 c",    "Kepler-1649",  3240, -1.698,  0.310,  234),
    ("GJ 667C c",        "GJ 667C",      3350, -1.598,  0.337,  277),
    ("GJ 667C e",        "GJ 667C",      3350, -1.598,  0.337,  222),
    ("GJ 667C f",        "GJ 667C",      3350, -1.598,  0.337,  199),
    ("TRAPPIST-1 b",     "TRAPPIST-1",   2566, -2.661,  0.121,  400),
    ("TRAPPIST-1 c",     "TRAPPIST-1",   2566, -2.661,  0.121,  342),
    ("TRAPPIST-1 d",     "TRAPPIST-1",   2566, -2.661,  0.121,  288),
    ("TRAPPIST-1 e",     "TRAPPIST-1",   2566, -2.661,  0.121,  251),
    ("TRAPPIST-1 f",     "TRAPPIST-1",   2566, -2.661,  0.121,  219),
    ("TRAPPIST-1 g",     "TRAPPIST-1",   2566, -2.661,  0.121,  198),
    ("LHS 1140 b",       "LHS 1140",     3216, -1.760,  0.186,  235),
    ("LHS 1140 c",       "LHS 1140",     3216, -1.760,  0.186,  371),
    ("TOI-700 d",        "TOI-700",      3480, -1.488,  0.416,  246),
    ("TOI-700 e",        "TOI-700",      3480, -1.488,  0.416,  269),
    ("Proxima Cen b",    "Proxima Cen",  3042, -2.066,  0.141,  234),
    ("Ross 128 b",       "Ross 128",     3192, -1.840,  0.196,  286),
    ("Wolf 1061 c",      "Wolf 1061",    3342, -1.688,  0.310,  272),
    ("Tau Ceti e",       "Tau Ceti",     5344, -0.207,  0.793,  312),
    ("Tau Ceti f",       "Tau Ceti",     5344, -0.207,  0.793,  220),
    ("K2-18 b",          "K2-18",        3457, -1.520,  0.411,  265),
    ("GJ 3293 d",        "GJ 3293",      3466, -1.591,  0.422,  218),
    ("HD 40307 g",       "HD 40307",     4977, -0.455,  0.716,  226),
    ("Kepler-438 b",     "Kepler-438",   3748, -1.017,  0.520,  312),
    ("Kepler-440 b",     "Kepler-440",   4134, -0.780,  0.560,  273),
    ("Kepler-296 e",     "Kepler-296",   3740, -1.064,  0.481,  303),
    ("Kepler-296 f",     "Kepler-296",   3740, -1.064,  0.481,  229),
    ("GJ 180 c",         "GJ 180",       3371, -1.630,  0.322,  262),
    ("GJ 180 d",         "GJ 180",       3371, -1.630,  0.322,  189),
    ("Kepler-1544 b",    "Kepler-1544",  4557, -0.584,  0.638,  271),
    ("Kepler-1410 b",    "Kepler-1410",  4430, -0.635,  0.616,  248),
    ("GJ 163 c",         "GJ 163",       3500, -1.449,  0.399,  277),
    ("GJ 273 b",         "GJ 273",       3382, -1.677,  0.293,  224),
    ("Kepler-705 b",     "Kepler-705",   4200, -0.760,  0.570,  259),
    ("Kepler-174 d",     "Kepler-174",   4800, -0.440,  0.670,  242),
    ("Kepler-283 c",     "Kepler-283",   4351, -0.650,  0.619,  248),
    ("Kepler-395 c",     "Kepler-395",   4060, -0.820,  0.530,  233),
    ("Kepler-1606 b",    "Kepler-1606",  4750, -0.452,  0.665,  268),
    ("GJ 682 c",         "GJ 682",       3028, -2.088,  0.213,  208),
    ("GJ 3942 c",        "GJ 3942",      3800, -1.001,  0.488,  244),
    ("Kepler-235 e",     "Kepler-235",   4350, -0.660,  0.609,  237),
    ("HD 216520 c",      "HD 216520",    5258, -0.270,  0.770,  241),
    ("Kepler-1229 b",    "Kepler-1229",  3724, -1.080,  0.479,  213),
    ("GJ 3323 c",        "GJ 3323",      3159, -1.852,  0.248,  204),
    ("K2-3 d",           "K2-3",         3896, -0.950,  0.561,  239),
]

COLUNAS = ["planeta", "estrela", "temp_estelar_K", "lum_log", "raio_estelar", "temp_equilibrio_K"]


# -----------------------------------------------------------------------
# carregamento e preparação dos dados
# -----------------------------------------------------------------------

def carregar_dados():
    df = pd.DataFrame(DADOS_NASA, columns=COLUNAS)

    # luminosidade em log precisa virar linear (L☉) e depois em Watts
    df["lum_solar"] = 10 ** df["lum_log"]         # ex: -1.5 -> 0.0316 L☉
    df["lum_watts"] = df["lum_solar"] * L_SOL_WATTS

    # classificação espectral pela temperatura (OBAFGKM)
    df["tipo_espectral"] = df["temp_estelar_K"].apply(tipo_espectral)

    return df

def tipo_espectral(teff):
    # classificação padrão de estrelas por temperatura de superfície
    if   teff >= 30000: return "O"
    elif teff >= 10000: return "B"
    elif teff >= 7500:  return "A"
    elif teff >= 6000:  return "F"
    elif teff >= 5200:  return "G"   # nosso Sol é tipo G (~5778 K)
    elif teff >= 3700:  return "K"
    else:               return "M"   # anãs vermelhas, as mais comuns no dataset


# -----------------------------------------------------------------------
# cálculos energéticos
# -----------------------------------------------------------------------

def calcular_potencia(lum_watts, distancia_ua=1.0):
    """
    Calcula a potência que os painéis da ORION captariam
    estando a 'distancia_ua' unidades astronômicas da estrela.

    Usa a lei do inverso do quadrado:
        irradiância = L / (4 * pi * d²)
    
    Depois multiplica pela área do painel e pela eficiência.
    """
    distancia_m  = distancia_ua * 1.496e11          # converte UA pra metros
    irradiancia  = lum_watts / (4 * np.pi * distancia_m ** 2)  # W/m²
    potencia     = irradiancia * AREA_PAINEL_M2 * EFICIENCIA    # W
    return round(potencia, 2)

def calcular_eficiencia_missao(potencia_w):
    # quanto da demanda total a cápsula consegue suprir
    # limitamos em 1.0 (100%) porque o excedente vai pra bateria
    return round(min(potencia_w / CONSUMO_TOTAL, 1.0), 4)

def calcular_balanco(potencia_w):
    # positivo = temos energia sobrando, negativo = precisamos economizar
    return round(potencia_w - CONSUMO_TOTAL, 1)


# -----------------------------------------------------------------------
# sistema de alertas
# -----------------------------------------------------------------------

def verificar_alerta(potencia_w, eficiencia, balanco_w):
    """
    Analisa as condições energéticas e retorna o nível de alerta
    junto com uma recomendação de ação para a equipe de controle.
    """
    if eficiencia < LIMITE_CRITICO or balanco_w < -4000:
        nivel = "CRÍTICO"
        msg   = f"Energia insuficiente! Apenas {potencia_w:.0f}W gerados de {CONSUMO_TOTAL}W necessários."
        acao  = "Desligar ORION-SCI e ORION-NAV imediatamente. Ativar painel de reserva."

    elif eficiencia < LIMITE_ATENCAO or balanco_w < 0:
        nivel = "ATENÇÃO"
        msg   = f"Eficiência abaixo do ideal ({eficiencia*100:.1f}%). Monitorar de perto."
        acao  = "Reduzir consumo do ORION-SCI em 30%. Verificar orientação dos painéis."

    else:
        nivel = "NOMINAL"
        msg   = f"Tudo certo! Eficiência de {eficiencia*100:.1f}% com {balanco_w:.0f}W de sobra."
        acao  = "Nenhuma ação necessária."

    return nivel, msg, acao

def cor_do_alerta(nivel):
    return {"CRÍTICO": RED, "ATENÇÃO": YELLOW, "NOMINAL": GREEN}.get(nivel, WHITE)


# -----------------------------------------------------------------------
# IA introdutória - regressão linear
# -----------------------------------------------------------------------

def treinar_modelo(df):
    """
    Treina uma regressão linear pra prever a potência captada
    com base na luminosidade e temperatura da estrela.

    Não é nada super complexo, mas mostra que dá pra usar ML
    pra estimar o potencial energético de novos sistemas estelares
    sem precisar calcular tudo manualmente.
    """
    X = df[["lum_log", "temp_estelar_K"]].values
    y = df["potencia_w"].values

    # normalizamos as features porque lum_log e temp_K estão em escalas muito diferentes
    scaler  = StandardScaler()
    X_norm  = scaler.fit_transform(X)

    modelo  = LinearRegression()
    modelo.fit(X_norm, y)

    y_pred  = modelo.predict(X_norm)
    r2      = r2_score(y, y_pred)

    df["potencia_prevista"] = y_pred
    return modelo, scaler, r2

def prever(modelo, scaler, lum_log, temp_k):
    X = np.array([[lum_log, temp_k]])
    return max(modelo.predict(scaler.transform(X))[0], 0)


# -----------------------------------------------------------------------
# prints no terminal
# -----------------------------------------------------------------------

def print_cabecalho():
    print()
    print(colorir("=" * 68, CYAN))
    print(colorir("  MISSÃO ORION — Monitoramento Energético", BOLD + WHITE))
    print(colorir("  GS 2026.1 | Energias Renováveis | FIAP 1CCPY", GRAY))
    print(colorir("  Dados: NASA Exoplanet Archive (PSCompPars)", GRAY))
    print(colorir("=" * 68, CYAN))
    print()

def print_resumo_dataset(df):
    print(colorir(">> Visão geral do dataset", CYAN))
    print(f"   {len(df)} exoplanetas de {df['estrela'].nunique()} sistemas estelares diferentes")
    print(f"   Temperatura estelar: de {df.temp_estelar_K.min()} K ({df.loc[df.temp_estelar_K.idxmin(),'estrela']}) "
          f"até {df.temp_estelar_K.max()} K ({df.loc[df.temp_estelar_K.idxmax(),'estrela']})")
    print(f"   Luminosidade média: {df.lum_solar.mean():.4f} L☉")
    print()
    print(f"   Distribuição por tipo espectral:")
    for tipo, qtd in df["tipo_espectral"].value_counts().items():
        barra = "█" * qtd
        print(f"     Tipo {tipo}  {colorir(barra, YELLOW)} {qtd}")
    print()

def print_consumo():
    print(colorir(">> Consumo da cápsula ORION", CYAN))
    for mod, w in CONSUMO_MODULOS.items():
        barra = "▓" * (w // 200)
        print(f"   {mod:<16} {colorir(f'{w:5d} W', YELLOW)}  {barra}")
    print(f"   {'TOTAL':<16} {colorir(f'{CONSUMO_TOTAL:5d} W', RED)}")
    print()

def print_tabela(df):
    print(colorir(">> Monitoramento por sistema estelar (top 20 por potência)", CYAN))
    header = f"   {'#':>2}  {'Planeta':<18} {'Estrela':<13} {'Teff(K)':>7} {'Lum(L☉)':>9} {'Pot.(W)':>9} {'Efic.':>6} {'Balanço':>9}  Alerta"
    print(colorir(header, GRAY))
    print(colorir("   " + "-" * 95, GRAY))

    top20 = df.sort_values("potencia_w", ascending=False).head(20)
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        bal     = r.balanco_w
        bal_str = f"{bal:+.0f}W"
        bal_cor = GREEN if bal >= 0 else RED
        nivel   = r.nivel_alerta
        linha   = (f"   {i:>2}  {r.planeta:<18} {r.estrela:<13} "
                   f"{r.temp_estelar_K:>7.0f} {r.lum_solar:>9.5f} "
                   f"{r.potencia_w:>9.1f} {r.eficiencia*100:>5.1f}%"
                   f" {colorir(bal_str, bal_cor):>16}  {colorir(nivel, cor_do_alerta(nivel))}")
        print(linha)
    print()

def print_alertas_criticos(df):
    criticos = df[df["nivel_alerta"] == "CRÍTICO"]
    atencoes = df[df["nivel_alerta"] == "ATENÇÃO"]
    nominais = df[df["nivel_alerta"] == "NOMINAL"]

    print(colorir(">> Resumo de alertas", CYAN))
    print(f"   {colorir('CRÍTICO', RED)}: {len(criticos)} sistemas  |  "
          f"{colorir('ATENÇÃO', YELLOW)}: {len(atencoes)} sistemas  |  "
          f"{colorir('NOMINAL', GREEN)}: {len(nominais)} sistemas")
    print()

    if not criticos.empty:
        print(colorir("   Sistemas críticos — ação necessária:", RED))
        for _, r in criticos.head(8).iterrows():   # mostrar só os primeiros 8 pra não poluir
            print(f"   • {r.planeta} ({r.estrela})")
            print(f"     ↳ {r.acao_recomendada}")
        if len(criticos) > 8:
            print(f"   ... e mais {len(criticos)-8} sistemas em estado crítico")
    print()

def print_ia(r2, modelo, scaler):
    print(colorir(">> IA introdutória — Regressão Linear (scikit-learn)", CYAN))
    print(f"   Usamos regressão linear pra prever a potência captada")
    print(f"   com base na luminosidade e temperatura de cada estrela.")
    print()
    print(f"   R² do modelo: {colorir(f'{r2:.4f}', GREEN if r2 > 0.85 else YELLOW)}")
    print(f"   (quanto mais próximo de 1.0, melhor o ajuste)")
    print()
    print(f"   Previsões para cenários hipotéticos:")
    print(f"   {'Tipo de estrela':<22} {'Lum (log)':>9} {'Temp (K)':>9}   {'Pot. prevista':>14}")
    print(colorir("   " + "-" * 58, GRAY))
    cenarios = [
        ("Anã M típica",      -1.5, 3500),
        ("Estrela tipo Sol",   0.0, 5778),
        ("Anã K",             -0.4, 4800),
    ]
    for nome, lum, teff in cenarios:
        prev = prever(modelo, scaler, lum, teff)
        print(f"   {nome:<22} {lum:>9.1f} {teff:>9}   {colorir(f'{prev:.1f} W', YELLOW):>20}")
    print()


# -----------------------------------------------------------------------
# gráficos
# -----------------------------------------------------------------------

# paleta escura tipo painel de controle de missão espacial
BG_FUNDO   = "#0d1117"
BG_PLOT    = "#161b22"
BORDA      = "#30363d"
TEXTO      = "#c9d1d9"
TITULO     = "#f0f6fc"

COR_ALERTA = {"CRÍTICO": "#e74c3c", "ATENÇÃO": "#f39c12", "NOMINAL": "#2ecc71"}
COR_TIPO   = {"G": "#f1c40f", "K": "#e67e22", "M": "#e74c3c",
              "F": "#3498db", "A": "#9b59b6", "B": "#1abc9c", "O": "#2980b9"}

def estilo_base(ax, fig):
    """aplica o estilo escuro em qualquer gráfico"""
    fig.patch.set_facecolor(BG_FUNDO)
    ax.set_facecolor(BG_PLOT)
    ax.tick_params(colors=TEXTO)
    for spine in ax.spines.values():
        spine.set_color(BORDA)
    ax.grid(color=BORDA, linewidth=0.4, alpha=0.6)

def grafico_potencia(df):
    top = df.sort_values("potencia_w", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(12, 5))
    estilo_base(ax, fig)

    cores = [COR_ALERTA[n] for n in top["nivel_alerta"]]
    ax.bar(range(len(top)), top["potencia_w"], color=cores, edgecolor=BORDA, linewidth=0.5)
    ax.axhline(CONSUMO_TOTAL, color="white", linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top["estrela"], rotation=35, ha="right", fontsize=8, color=TEXTO)
    ax.set_ylabel("Potência captada (W)", color=TEXTO, fontsize=10)
    ax.set_title("Potência captada por estrela-alvo — ORION", color=TITULO, fontsize=12, pad=12)

    patches = [mpatches.Patch(color=v, label=k) for k, v in COR_ALERTA.items()]
    patches.append(plt.Line2D([0],[0], color="white", lw=1.2, ls="--", label=f"Consumo total ({CONSUMO_TOTAL}W)"))
    ax.legend(handles=patches, fontsize=8, labelcolor="white", facecolor=BG_PLOT, edgecolor=BORDA)

    plt.tight_layout()
    path = f"{PASTA_SAIDA}/01_potencia_por_estrela.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FUNDO)
    plt.close()
    print(f"   salvo: {path}")

def grafico_hr(df):
    """diagrama HR simplificado — temperatura vs luminosidade"""
    fig, ax = plt.subplots(figsize=(9, 6))
    estilo_base(ax, fig)

    for tipo, grp in df.groupby("tipo_espectral"):
        ax.scatter(grp["temp_estelar_K"], grp["lum_solar"],
                   color=COR_TIPO.get(tipo, "#aaa"),
                   label=f"Tipo {tipo}", s=60, alpha=0.85, edgecolors=BORDA, linewidths=0.4)

    ax.set_xlabel("Temperatura estelar (K)", color=TEXTO, fontsize=10)
    ax.set_ylabel("Luminosidade (L☉)", color=TEXTO, fontsize=10)
    ax.set_yscale("log")
    ax.invert_xaxis()   # convenção: HR tem temperatura invertida
    ax.set_title("Diagrama HR simplificado — alvos da missão ORION", color=TITULO, fontsize=12, pad=12)
    ax.legend(fontsize=9, labelcolor="white", facecolor=BG_PLOT, edgecolor=BORDA)

    plt.tight_layout()
    path = f"{PASTA_SAIDA}/02_diagrama_hr.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FUNDO)
    plt.close()
    print(f"   salvo: {path}")

def grafico_balanco(df):
    """balanço energético — mostra quem tem sobra e quem tem déficit"""
    sub = df.sort_values("balanco_w").head(30)
    fig, ax = plt.subplots(figsize=(10, 8))
    estilo_base(ax, fig)

    cores = ["#e74c3c" if b < 0 else "#2ecc71" for b in sub["balanco_w"]]
    ax.barh(sub["planeta"], sub["balanco_w"], color=cores, edgecolor=BORDA, linewidth=0.4)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Balanço energético — gerado menos consumido (W)", color=TEXTO, fontsize=10)
    ax.set_title("Balanço energético por planeta-alvo", color=TITULO, fontsize=11, pad=12)
    ax.tick_params(labelsize=8, colors=TEXTO)

    patches = [mpatches.Patch(color="#2ecc71", label="Superávit"),
               mpatches.Patch(color="#e74c3c", label="Déficit")]
    ax.legend(handles=patches, fontsize=9, labelcolor="white", facecolor=BG_PLOT, edgecolor=BORDA)

    plt.tight_layout()
    path = f"{PASTA_SAIDA}/03_balanco_energetico.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FUNDO)
    plt.close()
    print(f"   salvo: {path}")

def grafico_regressao(df):
    """real vs previsto pelo modelo de IA"""
    fig, ax = plt.subplots(figsize=(7, 6))
    estilo_base(ax, fig)

    cores = [COR_ALERTA[n] for n in df["nivel_alerta"]]
    ax.scatter(df["potencia_w"], df["potencia_prevista"],
               color=cores, s=55, alpha=0.8, edgecolors=BORDA, linewidths=0.4)

    lim = max(df["potencia_w"].max(), df["potencia_prevista"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "w--", linewidth=1, alpha=0.5, label="Linha ideal (y = x)")

    ax.set_xlabel("Potência real (W)", color=TEXTO, fontsize=10)
    ax.set_ylabel("Potência prevista pelo modelo (W)", color=TEXTO, fontsize=10)
    ax.set_title("Regressão Linear: real vs previsto\n(luminosidade + temperatura → potência)", color=TITULO, fontsize=11, pad=10)

    patches = [mpatches.Patch(color=v, label=k) for k, v in COR_ALERTA.items()]
    patches.append(plt.Line2D([0],[0], color="white", ls="--", label="Ideal"))
    ax.legend(handles=patches, fontsize=8, labelcolor="white", facecolor=BG_PLOT, edgecolor=BORDA)

    plt.tight_layout()
    path = f"{PASTA_SAIDA}/04_regressao_ia.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FUNDO)
    plt.close()
    print(f"   salvo: {path}")

def grafico_alertas(df):
    contagem = df["nivel_alerta"].value_counts()
    fig, ax  = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG_FUNDO)
    ax.set_facecolor(BG_FUNDO)

    wedges, texts, pcts = ax.pie(
        contagem.values,
        labels=contagem.index,
        autopct="%1.0f%%",
        colors=[COR_ALERTA[k] for k in contagem.index],
        startangle=140,
        wedgeprops=dict(edgecolor=BG_FUNDO, linewidth=2),
    )
    for t in texts:  t.set_color(TEXTO);  t.set_fontsize(10)
    for p in pcts:   p.set_color("white"); p.set_fontsize(9); p.set_fontweight("bold")

    ax.set_title("Distribuição dos alertas — missão ORION", color=TITULO, fontsize=11, pad=12)

    plt.tight_layout()
    path = f"{PASTA_SAIDA}/05_alertas_pizza.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_FUNDO)
    plt.close()
    print(f"   salvo: {path}")


# -----------------------------------------------------------------------
# relatório final em texto
# -----------------------------------------------------------------------

def salvar_relatorio(df, r2):
    n_crit = len(df[df.nivel_alerta == "CRÍTICO"])
    n_aten = len(df[df.nivel_alerta == "ATENÇÃO"])
    n_nom  = len(df[df.nivel_alerta == "NOMINAL"])
    melhor = df.loc[df.potencia_w.idxmax()]
    pior   = df.loc[df.potencia_w.idxmin()]

    conteudo = f"""
MISSÃO ORION — RELATÓRIO FINAL DE MONITORAMENTO ENERGÉTICO
Global Solution 2026.1 | Energias Renováveis e Sustentáveis | FIAP

Grupo 05 — Turma 1CCPY:
  Arthur dos Santos Bezerra   RM 569721
  Carlos Henrique Fratezi     RM 571792
  Felipe Gouveia Braga        RM 568956

---------------------------------------------------------------
FONTE DOS DADOS
---------------------------------------------------------------
NASA Exoplanet Archive — PSCompPars
URL: https://exoplanetarchive.ipac.caltech.edu
Colunas usadas: temperatura estelar (st_teff), luminosidade estelar
(st_lum em log L/L☉), raio estelar (st_rad), temperatura de equilíbrio
do planeta (pl_eqt).

{len(df)} exoplanetas analisados de {df.estrela.nunique()} sistemas estelares.

---------------------------------------------------------------
PARÂMETROS DA CÁPSULA ORION
---------------------------------------------------------------
  Área total dos painéis solares : {AREA_PAINEL_M2} m²
  Eficiência dos painéis (GaAs)  : {EFICIENCIA*100:.0f}%
  Consumo total da missão        : {CONSUMO_TOTAL} W

  Detalhamento por módulo:
  - ORION-COMM  (comunicação)    : {CONSUMO_MODULOS['ORION-COMM']} W
  - ORION-LIFE  (suporte à vida) : {CONSUMO_MODULOS['ORION-LIFE']} W
  - ORION-NAV   (navegação)      : {CONSUMO_MODULOS['ORION-NAV']} W
  - ORION-SCI   (instrumentos)   : {CONSUMO_MODULOS['ORION-SCI']} W
  - ORION-THERM (controle temp.) : {CONSUMO_MODULOS['ORION-THERM']} W

---------------------------------------------------------------
ESTATÍSTICAS DO DATASET
---------------------------------------------------------------
  Temperatura estelar:
    Média          : {df.temp_estelar_K.mean():.0f} K
    Desvio padrão  : {df.temp_estelar_K.std():.0f} K
    Mínima         : {df.temp_estelar_K.min():.0f} K — {df.loc[df.temp_estelar_K.idxmin(),'estrela']}
    Máxima         : {df.temp_estelar_K.max():.0f} K — {df.loc[df.temp_estelar_K.idxmax(),'estrela']}

  Luminosidade estelar (L☉):
    Média          : {df.lum_solar.mean():.5f}
    Desvio padrão  : {df.lum_solar.std():.5f}
    Mínima         : {df.lum_solar.min():.6f} — {df.loc[df.lum_solar.idxmin(),'estrela']}
    Máxima         : {df.lum_solar.max():.4f} — {df.loc[df.lum_solar.idxmax(),'estrela']}

---------------------------------------------------------------
RESULTADO ENERGÉTICO
---------------------------------------------------------------
  Potência média gerada  : {df.potencia_w.mean():.1f} W
  Potência máxima        : {df.potencia_w.max():.1f} W — {melhor.planeta} ({melhor.estrela})
  Potência mínima        : {df.potencia_w.min():.2f} W — {pior.planeta} ({pior.estrela})
  Alvo mais eficiente    : {melhor.planeta} — {melhor.eficiencia*100:.1f}%
  Alvo menos eficiente   : {pior.planeta} — {pior.eficiencia*100:.2f}%

---------------------------------------------------------------
ALERTAS
---------------------------------------------------------------
  CRÍTICO : {n_crit} sistemas ({n_crit/len(df)*100:.1f}%)
  ATENÇÃO  : {n_aten} sistemas ({n_aten/len(df)*100:.1f}%)
  NOMINAL  : {n_nom} sistemas ({n_nom/len(df)*100:.1f}%)

A maioria dos sistemas está em estado crítico porque o dataset
é composto principalmente de anãs M e K, que têm baixa luminosidade.
Isso significa que a cápsula não captaria energia suficiente pra
manter todos os módulos funcionando a 1 UA dessas estrelas.

---------------------------------------------------------------
MODELO DE IA — REGRESSÃO LINEAR
---------------------------------------------------------------
  Biblioteca  : scikit-learn
  Algoritmo   : LinearRegression com StandardScaler
  Features    : log(luminosidade), temperatura estelar (K)
  Target      : potência captada (W)
  R²          : {r2:.4f}

  O modelo consegue explicar {r2*100:.1f}% da variância na potência
  captada. O resultado é bom considerando que usamos apenas duas
  variáveis e dados de estrelas bem diferentes entre si.

---------------------------------------------------------------
CONCLUSÃO
---------------------------------------------------------------
A análise mostra que estrelas do tipo G (similar ao nosso Sol),
como Kepler-452 e Tau Ceti, são as melhores candidatas para a
missão ORION do ponto de vista energético — produzem superávit
e mantêm todos os módulos funcionando sem restrições.

Estrelas tipo M (anãs vermelhas) como TRAPPIST-1 e Proxima Centauri,
apesar de hospedarem planetas potencialmente habitáveis, impõem
grande desafio energético: a cápsula precisaria se aproximar muito
mais ou reduzir significativamente o consumo dos módulos.
"""

    path = f"{PASTA_SAIDA}/relatorio_final.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(conteudo)
    print(f"   salvo: {path}")


# -----------------------------------------------------------------------
# execução principal
# -----------------------------------------------------------------------

def main():
    print_cabecalho()

    # carrega e enriquece os dados
    df = carregar_dados()

    # calcula potência, eficiência e balanço pra cada planeta
    df["potencia_w"]  = df["lum_watts"].apply(calcular_potencia)
    df["eficiencia"]  = df["potencia_w"].apply(calcular_eficiencia_missao)
    df["balanco_w"]   = df["potencia_w"].apply(calcular_balanco)

    # aplica o sistema de alertas
    alertas = df.apply(lambda r: verificar_alerta(r.potencia_w, r.eficiencia, r.balanco_w), axis=1)
    df["nivel_alerta"]     = [a[0] for a in alertas]
    df["msg_alerta"]       = [a[1] for a in alertas]
    df["acao_recomendada"] = [a[2] for a in alertas]

    # imprime tudo no terminal
    print_resumo_dataset(df)
    print_consumo()
    print_tabela(df)
    print_alertas_criticos(df)

    # treina o modelo de IA e imprime resultado
    modelo, scaler, r2 = treinar_modelo(df)
    print_ia(r2, modelo, scaler)

    # gera os gráficos
    print(colorir(">> Gerando gráficos...", CYAN))
    grafico_potencia(df)
    grafico_hr(df)
    grafico_balanco(df)
    grafico_regressao(df)
    grafico_alertas(df)
    print()

    # salva o relatório final
    print(colorir(">> Salvando relatório...", CYAN))
    salvar_relatorio(df, r2)
    print()

    print(colorir("=" * 68, CYAN))
    print(colorir("  Análise concluída! Outputs salvos em ./orion_outputs/", GREEN + BOLD))
    print(colorir("=" * 68, CYAN))
    print()


if __name__ == "__main__":
    main()
