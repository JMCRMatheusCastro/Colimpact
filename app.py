"""
ColImpact (Column Impact Analysis Tool) - versão Streamlit

Programador: José Matheus de Castro Rodrigues  |  Última versão do notebook: 22/09/2026
Recife / PE, Brasil.

Como executar:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st

import colimpact_core as core
import colimpact_plots as plots

st.set_page_config(page_title="ColImpact", page_icon="🏛️", layout="wide")

N_MAX_CAMADAS = 8
OPCOES_BITOLA = [float(x) for x in core.BITOLAS]
OPCOES_TIPO = ["Retangular", "Circular"]
OPCOES_APOIO = ["Biapoiado", "Engastado (em balanço)"]

# --------------------------------------------------------------------------------------
# Exemplos do notebook (seção 1)
# --------------------------------------------------------------------------------------
EXEMPLOS = {
    "Ex. 1 – Seção retangular, biapoiado (com impacto)": dict(
        tipo="Retangular", apoio="Biapoiado",
        b=0.30, h=0.30, L=3.4, dLinha=0.0413, n_camadas=2, n_0=2, n_1=2, bitola=20.0,
        fck=30.0, fyk=500.0, Es=210.0,
        Nd=1127.9, qd=-5.32, MBd=-10.8, MTd=10.0, QTd=0.0, Fveic=50.0, Pos=0.5,
    ),
    "Ex. 2 – Seção circular, em balanço": dict(
        tipo="Circular", apoio="Engastado (em balanço)",
        D=0.40, L=4.0, dLinha=0.0475, n_barras_circ=8, bitola=25.0,
        fck=30.0, fyk=500.0, Es=210.0,
        Nd=300.0, qd=0.0, MBd=0.0, MTd=0.0, QTd=30.0, Fveic=0.0, Pos=0.5,
    ),
}
PERSONALIZADO = "Personalizado"

DEFAULTS_GERAIS = dict(
    b=0.30, h=0.30, D=0.40, L=3.4, dLinha=0.04, n_camadas=2, n_barras_circ=8, bitola=20.0,
    fck=30.0, fyk=500.0, Es=210.0, Nd=0.0, qd=0.0, MBd=0.0, MTd=0.0, QTd=0.0, Fveic=0.0, Pos=0.5,
    DeltaZ=0.05, tol=1e-4, max_iter=100, lang="Português", fsi_lim=35.0,
)

ss = st.session_state


def aplicar_exemplo():
    """Preenche os campos com um dos exemplos do notebook."""
    nome = ss["exemplo"]
    if nome not in EXEMPLOS:
        return
    ex = EXEMPLOS[nome]
    for k, v in ex.items():
        ss[k] = v
    for i in range(N_MAX_CAMADAS):
        ss[f"n_{i}"] = ex.get(f"n_{i}", 2)


if "iniciado" not in ss:
    for k, v in DEFAULTS_GERAIS.items():
        ss[k] = v
    ss["exemplo"] = list(EXEMPLOS)[0]
    aplicar_exemplo()
    ss["iniciado"] = True

# O Streamlit apaga o estado de widgets que não são desenhados numa execução (ex.: b e h quando a seção é
# circular). Reatribuir no início de cada execução mantém os valores já digitados ao alternar as opções.
for _k in list(DEFAULTS_GERAIS) + [f"n_{i}" for i in range(N_MAX_CAMADAS)]:
    if _k in ss:
        ss[_k] = ss[_k]


def val(chave):
    """Valor atual de um campo, com o padrão como reserva."""
    return ss[chave] if chave in ss else DEFAULTS_GERAIS.get(chave, 2)


# --------------------------------------------------------------------------------------
# Cabeçalho
# --------------------------------------------------------------------------------------
st.title("🏛️ ColImpact")
st.markdown("**Column Impact Analysis Tool** — análise não linear de pilares de concreto armado sujeitos a "
            "carregamentos laterais, incluindo impacto de veículos por Forças Estáticas Equivalentes (FEE).")
st.caption("Programador: José Matheus de Castro Rodrigues · Última versão do código: 22/09/2026 · Recife / PE, Brasil.")

with st.expander("Convenções de sinais, hipóteses e referências"):
    st.markdown(
        """
**Convenções de sinais**

- **Nd**: (+) compressão  ·  **qd**: (+) para a direita  ·  **Fveic**: (+) para a direita
- **MBd** (momento na base): (+) sentido horário  ·  **MTd** (momento no topo): (+) sentido anti-horário
- **QTd** (força transversal no topo, só pilar em balanço): (+) para a direita

**Hipóteses do modelo**

- Não linearidade geométrica (NLG) por diferenças finitas e não linearidade física (NLF) por rigidez secante
  iterativa, a partir do diagrama N, M, 1/r.
- Esforços locais de 2ª ordem calculados com as cargas divididas por γf3 e depois majorados por γf3 = 1,1
  (ABNT NBR 6118, item 15.3.1).
- γc = 1,2 e γs = 1,0 (fixos, como no código original). Fator de Segurança de Impacto: FSI = Mrd / Msd.

**Referências**

- ARAÚJO, J. M. *Curso de Concreto Armado*, Vol. 1 e Vol. 3. 4. ed. Rio Grande: Dunas, 2014.
- ABNT NBR 6118:2023 — Projeto de estruturas de concreto. Rio de Janeiro: ABNT, 2023.
- FERREIRA, P. F. S. *Estacas de Seção Circular Semienterradas Submetidas a Cargas Laterais*. Dissertação (Mestrado), UFPE, Recife.
- GHALI, A.; NEVILLE, A. *Structural Analysis: A unified classical and matrix approach*. 7. ed. CRC Press, 2017.
- ZIENKIEWICZ, O. C.; MORGAN, K. *Finite Elements and Approximation*. Dover Publications, 2006.
        """
    )

# --------------------------------------------------------------------------------------
# Barra lateral - dados de entrada
# --------------------------------------------------------------------------------------
sb = st.sidebar
sb.header("Dados de entrada")

sb.selectbox("Carregar exemplo do notebook", list(EXEMPLOS) + [PERSONALIZADO], key="exemplo",
             on_change=aplicar_exemplo)

sb.subheader("1. Seção e vinculação")
sb.radio("Tipo de seção", OPCOES_TIPO, key="tipo", horizontal=True)
sb.radio("Vinculação", OPCOES_APOIO, key="apoio")

sb.subheader("2. Geometria")
if ss["tipo"] == "Retangular":
    c1, c2 = sb.columns(2)
    c1.number_input("b (m)", min_value=0.05, step=0.01, format="%.3f", key="b")
    c2.number_input("h (m)", min_value=0.05, step=0.01, format="%.3f", key="h")
else:
    sb.number_input("D (m)", min_value=0.05, step=0.01, format="%.3f", key="D")
sb.number_input("L – comprimento do pilar (m)", min_value=0.5, step=0.1, format="%.2f", key="L")

sb.subheader("3. Armadura longitudinal")
sb.number_input("d′ – distância do CG da armadura à face (m)", min_value=0.01, step=0.001, format="%.4f", key="dLinha")
sb.selectbox("Bitola (mm)", OPCOES_BITOLA, key="bitola")
if ss["tipo"] == "Retangular":
    sb.number_input("Número de camadas de armadura", min_value=2, max_value=N_MAX_CAMADAS, step=1, key="n_camadas",
                    help="Camadas igualmente espaçadas entre as coordenadas −h/2+d′ e h/2−d′.")
    cols = sb.columns(min(int(ss["n_camadas"]), 4))
    for i in range(int(ss["n_camadas"])):
        cols[i % len(cols)].number_input(f"Barras – camada {i + 1}", min_value=1, max_value=50, step=1, key=f"n_{i}")
else:
    sb.number_input("Número de barras (par, ≥ 4)", min_value=4, max_value=64, step=2, key="n_barras_circ",
                    help="Barras igualmente espaçadas em um círculo de raio D/2 − d′.")

sb.subheader("4. Materiais")
c1, c2, c3 = sb.columns(3)
c1.number_input("fck (MPa)", min_value=10.0, step=5.0, format="%.1f", key="fck")
c2.number_input("fyk (MPa)", min_value=100.0, step=50.0, format="%.1f", key="fyk")
c3.number_input("Es (GPa)", min_value=100.0, step=10.0, format="%.0f", key="Es")

sb.subheader("5. Carregamentos de cálculo")
sb.number_input("Nd (kN) – normal, (+) compressão", step=10.0, format="%.2f", key="Nd")
sb.number_input("qd (kN/m) – distribuído, (+) para a direita", step=0.5, format="%.2f", key="qd")
if ss["apoio"] == OPCOES_APOIO[0]:
    sb.number_input("MBd (kN·m) – momento na base, (+) horário", step=1.0, format="%.2f", key="MBd")
sb.number_input("MTd (kN·m) – momento no topo, (+) anti-horário", step=1.0, format="%.2f", key="MTd")
if ss["apoio"] == OPCOES_APOIO[1]:
    sb.number_input("QTd (kN) – força transversal no topo, (+) para a direita", step=1.0, format="%.2f", key="QTd")
sb.number_input("Fveic (kN) – impacto do veículo, (+) para a direita", step=5.0, format="%.2f", key="Fveic",
                help="Use 0 para analisar o pilar sem impacto.")
sb.number_input("Posição do impacto (m, a partir da base)", min_value=0.0, step=0.1, format="%.2f", key="Pos",
                disabled=(ss["Fveic"] == 0),
                help="Aplicada no nó mais próximo da malha de ΔZ. Recomenda-se múltiplos de 0,1 m.")

with sb.expander("Opções avançadas e gráficos"):
    st.number_input("ΔZ – discretização longitudinal (m)", min_value=0.01, max_value=0.25, step=0.01,
                    format="%.3f", key="DeltaZ", help="L precisa ser múltiplo de ΔZ.")
    st.number_input("Tolerância da iteração não linear", min_value=1e-8, max_value=1e-1, step=1e-5,
                    format="%.1e", key="tol")
    st.number_input("Máximo de iterações", min_value=5, max_value=500, step=5, key="max_iter")
    st.selectbox("Idioma dos gráficos", ["Português", "English"], key="lang")
    st.number_input("Limite do eixo do FSI no gráfico", min_value=5.0, max_value=200.0, step=5.0,
                    format="%.0f", key="fsi_lim")


# --------------------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------------------
def coletar_parametros():
    retangular = ss["tipo"] == "Retangular"
    p = dict(
        tipo_secao="retangular" if retangular else "circular",
        apoio="biapoiado" if ss["apoio"] == OPCOES_APOIO[0] else "engastado",
        L=float(val("L")), DeltaZ=float(val("DeltaZ")), dLinha=float(val("dLinha")), bitola=float(val("bitola")),
        fck=float(val("fck")), fyk=float(val("fyk")), Es=float(val("Es")),
        Nd=float(val("Nd")), qd=float(val("qd")), MBd=float(val("MBd")), MTd=float(val("MTd")),
        QTd=float(val("QTd")), Fveic=float(val("Fveic")),
        Pos_Fveic=float(val("Pos")) if float(val("Fveic")) != 0 else None,
    )
    if retangular:
        p.update(b=float(val("b")), h=float(val("h")),
                 n_barras=[int(val(f"n_{i}")) for i in range(int(val("n_camadas")))])
    else:
        p.update(D=float(val("D")), n_barras_circ=int(val("n_barras_circ")))
    return p


@st.cache_data(show_spinner=False, max_entries=20)
def rodar_analise(p_json, tol, max_iter):
    return core.executar_analise(json.loads(p_json), tol=tol, max_iter=max_iter)


@st.cache_data(show_spinner=False, max_entries=60)
def png_figura(chave, nome, lang, fsi_lim, _r):
    """PNG em 600 dpi (cacheado) para o botão de download."""
    return plots.fig_to_png(montar_figura(nome, _r, lang, fsi_lim, dpi=600), dpi=600)


def montar_figura(nome, r, lang, fsi_lim, dpi=150):
    if nome == "mchi":
        return plots.plot_Momento_curvatura(r, lang, dpi)
    if nome == "mei":
        return plots.plot_Momento_vs_Rigidez(r, lang, dpi)
    if nome == "desl":
        return plots.plot_deslocamentos(r, lang, dpi)
    if nome == "mom":
        return plots.plot_momentos_fletores(r, lang, dpi)
    if nome == "fsi":
        return plots.plot_fator_seguranca_final(r, lang, limite_escala=fsi_lim, dpi=dpi)
    raise ValueError(nome)


def mostrar_figura(nome, titulo_arquivo, r, lang, fsi_lim, chave):
    st.pyplot(montar_figura(nome, r, lang, fsi_lim))
    st.download_button("⬇️ Baixar figura (PNG, 600 dpi)", png_figura(chave, nome, lang, fsi_lim, r),
                       file_name=f"ColImpact_{titulo_arquivo}.png", mime="image/png", key=f"dl_{nome}")


def csv_bytes(df):
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")   # abre direto no Excel pt-BR


# --------------------------------------------------------------------------------------
# Botão e cálculo
# --------------------------------------------------------------------------------------
params = coletar_parametros()
chave_atual = json.dumps(params, sort_keys=True) + f"|{val('tol')}|{val('max_iter')}"

if st.button("▶ Calcular", type="primary"):
    erros = core.validar_entradas(params)
    if erros:
        for e in erros:
            st.error(e)
    else:
        try:
            with st.spinner("Calculando: diagrama N, M, 1/r e iteração da rigidez não linear..."):
                resultado = rodar_analise(json.dumps(params, sort_keys=True), float(val("tol")), int(val("max_iter")))
            ss["resultado"] = dict(chave=chave_atual, r=resultado)
        except Exception as exc:
            st.error(f"Não foi possível concluir a análise: {exc}")
            with st.expander("Detalhes técnicos"):
                st.code(traceback.format_exc())

if "resultado" not in ss:
    st.info("Confira os dados na barra lateral (já vêm preenchidos com o Ex. 1 do notebook) e clique em **Calcular**.")
    st.stop()

if ss["resultado"]["chave"] != chave_atual:
    st.warning("Os dados de entrada mudaram desde o último cálculo. Clique em **Calcular** para atualizar os resultados.")

# --------------------------------------------------------------------------------------
# Resultados
# --------------------------------------------------------------------------------------
r = ss["resultado"]["r"]
chave_res = ss["resultado"]["chave"]
lang = "pt" if val("lang") == "Português" else "en"
fsi_lim = float(val("fsi_lim"))
g3 = r["gamma_f3"]

for aviso in r["avisos"]:
    st.warning(aviso)

st.header("Resultados")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Momento resistente Mrd", f"{r['Mrd']:.2f} kN·m")
m2.metric("Msd máx. (NLG & NLF)", f"{np.max(np.abs(r['M_NLG_NLF_maj'])):.2f} kN·m")
m3.metric("FSI mínimo", f"{r['fsi_min']:.2f}", help=f"Ocorre em z = {r['z_critico']:.2f} m (a partir da base).")
m4.metric("Desl. máx. (NLG & NLF)", f"{np.max(np.abs(r['W_NLG_NLF'])) * 1000:.2f} mm")
m5.metric("Iterações (rigidez NL)", f"{len(r['historico'])}", help="Convergiu." if r["convergiu"] else "Não convergiu.")

if r["fsi_min"] < 1.0:
    st.error(f"FSI mínimo = {r['fsi_min']:.2f} < 1,0 em z = {r['z_critico']:.2f} m: "
             "o momento solicitante supera o resistente nessa seção.")
else:
    st.success(f"FSI mínimo = {r['fsi_min']:.2f} (≥ 1,0), em z = {r['z_critico']:.2f} m.")

abas = st.tabs(["Resumo", "N, M, 1/r", "M × EI", "Deslocamentos", "Momentos fletores", "FSI", "Exportar"])

# ---- Resumo -------------------------------------------------------------------------
with abas[0]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Armadura e seção")
        df_arm = pd.DataFrame({
            "Camada": np.arange(1, len(r["ds"]) + 1),
            "y (m)": np.round(r["ds"], 4),
            "Barras": r["n_camadas"].astype(int),
            "As (cm²)": np.round(r["As_camadas"] * 1e4, 3),
        })
        st.dataframe(df_arm, hide_index=True)
        st.write(f"**As total:** {r['As_total_cm2']:.2f} cm²")
        st.write(f"**Linha neutra (x) na ruptura:** {r['x_raiz']:.4f} m  ·  "
                 f"**Md (NLG):** {r['Md']:.2f} kN·m  ·  **Mr (fissuração):** {r['Mr']:.2f} kN·m")
    with c2:
        st.subheader("Convergência da rigidez não linear")
        df_it = pd.DataFrame({"Iteração": np.arange(1, len(r["historico"]) + 1), "Resíduo": r["historico"]})
        st.dataframe(df_it.style.format({"Resíduo": "{:.5f}"}), hide_index=True)

    st.subheader("Deslocamentos e momentos máximos")
    st.dataframe(core.tabela_deslocamentos(plots.series_deslocamentos(r, lang)).style.format(
        {"Desl. Máx. (m)": "{:.6f}", "Desl. Mín. (m)": "{:.6f}"}), hide_index=True)
    st.dataframe(core.tabela_momentos_fletores(plots.series_momentos(r, lang)).style.format(
        {"MF Máx. (kN.m)": "{:.4f}", "MF Mín. (kN.m)": "{:.4f}"}), hide_index=True)

# ---- N, M, 1/r ----------------------------------------------------------------------
with abas[1]:
    st.caption("Curva com 0,85 ηc fcd e curva com 1,1 fcd (Nd/1,1), usada na rigidez secante.")
    mostrar_figura("mchi", "Momento_Curvatura", r, lang, fsi_lim, chave_res)

# ---- M x EI -------------------------------------------------------------------------
with abas[2]:
    mostrar_figura("mei", "Momento_Rigidez", r, lang, fsi_lim, chave_res)

# ---- Deslocamentos ------------------------------------------------------------------
with abas[3]:
    mostrar_figura("desl", "Deslocamentos_Pilar", r, lang, fsi_lim, chave_res)

# ---- Momentos fletores --------------------------------------------------------------
with abas[4]:
    st.caption(f"Momentos NLG & NLF já majorados por γf3 = {g3:.1f} (NBR 6118, item 15.3.1).")
    mostrar_figura("mom", "Momentos_Fletores_Pilar", r, lang, fsi_lim, chave_res)

# ---- FSI ----------------------------------------------------------------------------
with abas[5]:
    st.caption(f"FSI = Mrd / Msd, com Mrd = {r['Mrd']:.2f} kN·m e Msd = γf3 · M(NLG & NLF).")
    mostrar_figura("fsi", "Fator_Seguranca_Impacto", r, lang, fsi_lim, chave_res)
    st.subheader("Tabela do FSI ao longo do pilar (topo → base)")
    st.dataframe(core.tabela_fs(r["M_NLG_NLF_maj"], r["Mrd"]), hide_index=True, height=400)

# ---- Exportar -----------------------------------------------------------------------
with abas[6]:
    df_pilar = pd.DataFrame({
        "z (m)": r["z"],
        "W_LG&LF (m)": r["W_P0"].flatten(),
        "W_NLG&LF (m)": r["W_NLG"].flatten(),
        "W_NLG&NLF (m)": r["W_NLG_NLF"].flatten(),
        "M_LG&LF (kN.m)": r["M_P0"],
        "M_NLG&LF (kN.m)": r["M_NLG"],
        "M_NLG&NLF majorado (kN.m)": r["M_NLG_NLF_maj"],
        "EI linear (kN.m2)": r["EIef"][0] * np.ones_like(r["M_P0"]),
        "EI nao linear (kN.m2)": r["EIef_NL"],
        "FSI": r["fsi"],
    })
    n_max = max(len(r["chi85"]), len(r["chi11"]))
    df_curva = pd.DataFrame({
        "chi 0.85fcd (1/m)": np.pad(r["chi85"], (0, n_max - len(r["chi85"])), constant_values=np.nan),
        "M 0.85fcd (kN.m)": np.pad(r["M85"], (0, n_max - len(r["M85"])), constant_values=np.nan),
        "chi 1.1fcd (1/m)": np.pad(r["chi11"], (0, n_max - len(r["chi11"])), constant_values=np.nan),
        "M 1.1fcd (kN.m)": np.pad(r["M11"], (0, n_max - len(r["M11"])), constant_values=np.nan),
    })

    st.write("Arquivos CSV com separador `;` e vírgula decimal (abrem direto no Excel em português).")
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Resultados ao longo do pilar (CSV)", csv_bytes(df_pilar),
                       file_name="ColImpact_resultados_pilar.csv", mime="text/csv")
    d2.download_button("⬇️ Curva N, M, 1/r (CSV)", csv_bytes(df_curva),
                       file_name="ColImpact_curva_N_M_1r.csv", mime="text/csv")
    st.dataframe(df_pilar, hide_index=True, height=350)
