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
import colimpact_modelo as mod
import colimpact_plots as plots

st.set_page_config(page_title="ColImpact", page_icon="🏛️", layout="wide")

N_MAX_CAMADAS = 8
OPCOES_BITOLA = [float(x) for x in core.BITOLAS]
OPCOES_ESTRIBO = [5.0, 6.3, 8.0, 10.0, 12.5]
OPCOES_TIPO = ["Retangular", "Circular"]
OPCOES_APOIO = ["Biapoiado", "Engastado (em balanço)"]
OPCOES_ARM = ["Por faces (cantos + barras extras)", "Por camadas (avançado)"]
MODOS_IMPACTO = ["Sem impacto", "Garagem – NBR 6120, Tabela 13", "Via pública/particular – NBR 6120, Tabela 14",
                 "Livre (definir F e altura)"]
OPCOES_DIR = ["Fx – paralela ao fluxo dos veículos", "Fy – perpendicular ao fluxo dos veículos"]
OPCOES_SENT = ["→ para a direita (+)", "← para a esquerda (−)"]

# --------------------------------------------------------------------------------------
# Valores iniciais (o app abre com um modelo completo para o desenho já aparecer)
# --------------------------------------------------------------------------------------
DEFAULTS_GERAIS = dict(
    tipo=OPCOES_TIPO[0], apoio=OPCOES_APOIO[0],
    b=0.30, h=0.30, D=0.40, L=3.4, bitola=20.0, cobrimento=25.0, estribo=6.3, dLinha=0.0413, dLinha_manual=False,
    arm_modo=OPCOES_ARM[0], n_face_h=0, n_face_b=0, n_camadas=2, n_barras_circ=8,
    fck=30.0, fyk=500.0, Es=210.0, Nd=1127.9, qd=-5.32, MBd=-10.8, MTd=10.0, QTd=0.0,
    imp_modo=MODOS_IMPACTO[1], imp_cat="I", imp_caso="sem_protecao", imp_dist=0.0, imp_dir=OPCOES_DIR[1],
    imp_sent=OPCOES_SENT[0], imp_rampa=False, imp_F_livre=50.0, imp_H_livre=0.5,
    DeltaZ=0.05, tol=1e-4, max_iter=100, lang="Português", fsi_lim=35.0,
)

ss = st.session_state


if "iniciado" not in ss:
    for k, v in DEFAULTS_GERAIS.items():
        ss[k] = v
    for i in range(N_MAX_CAMADAS):
        ss[f"n_{i}"] = 2
    ss["iniciado"] = True

# O Streamlit apaga o estado de widgets que não são desenhados numa execução (ex.: b e h quando a seção é
# circular). Reatribuir no início de cada execução mantém os valores já digitados ao alternar as opções.
for _k in list(DEFAULTS_GERAIS) + [f"n_{i}" for i in range(N_MAX_CAMADAS)]:
    if _k in ss:
        ss[_k] = ss[_k]


def val(chave):
    """Valor atual de um campo, com o padrão como reserva."""
    return ss[chave] if chave in ss else DEFAULTS_GERAIS.get(chave, 2)


def dLinha_atual():
    """d' (m): automático (cobrimento + estribo + φ/2) ou o valor digitado."""
    if val("dLinha_manual"):
        return float(val("dLinha"))
    return mod.dLinha_de_cobrimento(val("cobrimento"), val("estribo"), val("bitola"))


def lista_n_barras():
    """Barras por camada (seção retangular), a partir do modo de definição escolhido."""
    if val("arm_modo") == OPCOES_ARM[0]:
        return mod.camadas_por_faces(val("n_face_h"), val("n_face_b"))
    return [int(val(f"n_{i}")) for i in range(int(val("n_camadas")))]


def impacto_atual():
    """Força de impacto (com sinal), altura de aplicação e textos de apoio, conforme o modo escolhido."""
    modo = val("imp_modo")
    if modo == MODOS_IMPACTO[0]:
        return dict(F=0.0, H=0.0, F_label="", resumo="Sem impacto de veículo.")
    if modo == MODOS_IMPACTO[3]:
        F, H = float(val("imp_F_livre")), float(val("imp_H_livre"))
        return dict(F=F, H=H, F_label=r"$F_{\mathrm{veic}}$",
                    resumo=f"Impacto livre: F = {abs(F):.2f} kN ({'→' if F >= 0 else '←'}) a {H:.2f} m da base.")
    fonte = "garagem" if modo == MODOS_IMPACTO[1] else "via"
    chave = val("imp_cat") if fonte == "garagem" else val("imp_caso")
    direcao = "x" if val("imp_dir") == OPCOES_DIR[0] else "y"
    sentido = 1 if val("imp_sent") == OPCOES_SENT[0] else -1
    r = mod.impacto_nbr6120(fonte, chave, direcao, sentido, bool(val("imp_rampa")), float(val("imp_dist")))
    extra = ""
    if r["fator"] != 1.0:
        extra = f" (tabela {r['F_tabela']:g} kN × {r['fator']:g})"
    return dict(F=r["F"], H=r["H"], F_label=rf"$F_{{\mathrm{{veic}},{direcao}}}$",
                resumo=f"F{direcao} = {r['F_abs']:g} kN{extra} a {r['H']:.2f} m da base, "
                       f"{'→' if sentido > 0 else '←'}.")


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
**Convenções de sinais** (o desenho do modelo, logo abaixo, mostra o sentido real de cada carga)

- **Nd**: (+) compressão  ·  **qd**: (+) para a direita  ·  **F de impacto**: (+) para a direita
- **MBd** (momento na base): (+) sentido horário  ·  **MTd** (momento no topo): (+) sentido anti-horário
- **QTd** (força transversal no topo, só pilar em balanço): (+) para a direita

**Impacto de veículos (ABNT NBR 6120:2019)**

- Categorias das Tabelas 13 (garagens) e 14 (pilares adjacentes a vias). São ações excepcionais (NBR 8681);
  Fx (paralela ao fluxo) e Fy (perpendicular) **não atuam ao mesmo tempo**: analise uma direção por vez e, para a
  outra, gire a seção (troque b e h).
- A força é aplicada como carga concentrada no nó da malha mais próximo da altura H (a norma admite atuação em uma
  faixa de 25 cm de altura). Perto de descidas de rampa, as forças da Tabela 13 dobram; na Tabela 14 elas decrescem
  linearmente com a distância do pilar à via, até zero a 10 m.

**Hipóteses do modelo**

- Não linearidade geométrica (NLG) por diferenças finitas e não linearidade física (NLF) por rigidez secante
  iterativa, a partir do diagrama N, M, 1/r.
- Esforços locais de 2ª ordem calculados com as cargas divididas por γf3 e depois majorados por γf3 = 1,1
  (ABNT NBR 6118, item 15.3.1).
- γc = 1,2 e γs = 1,0 (fixos, como no código original). Fator de Segurança de Impacto: FSI = Mrd / Msd.

**Referências**

- ARAÚJO, J. M. *Curso de Concreto Armado*, Vol. 1 e Vol. 3. 4. ed. Rio Grande: Dunas, 2014.
- ABNT NBR 6118:2023 — Projeto de estruturas de concreto. Rio de Janeiro: ABNT, 2023.
- ABNT NBR 6120:2019 — Ações para o cálculo de estruturas de edificações. Rio de Janeiro: ABNT, 2019.
- FERREIRA, P. F. S. *Estacas de Seção Circular Semienterradas Submetidas a Cargas Laterais*. Dissertação (Mestrado), UFPE, Recife.
- GHALI, A.; NEVILLE, A. *Structural Analysis: A unified classical and matrix approach*. 7. ed. CRC Press, 2017.
- ZIENKIEWICZ, O. C.; MORGAN, K. *Finite Elements and Approximation*. Dover Publications, 2006.
        """
    )

# Espaço reservado para a pré-visualização: é preenchido depois da barra lateral (que define os valores),
# mas aparece aqui, no topo da página, redesenhado a cada alteração.
area_modelo = st.container()

# --------------------------------------------------------------------------------------
# Barra lateral - dados de entrada
# --------------------------------------------------------------------------------------
sb = st.sidebar
sb.header("Dados de entrada")

sb.subheader("1. Seção e vinculação")
sb.radio("Tipo de seção", OPCOES_TIPO, key="tipo", horizontal=True)
sb.radio("Vinculação", OPCOES_APOIO, key="apoio")

sb.subheader("2. Geometria")
if ss["tipo"] == "Retangular":
    c1, c2 = sb.columns(2)
    c1.number_input("b (m)", min_value=0.05, step=0.01, format="%.3f", key="b")
    c2.number_input("h (m)", min_value=0.05, step=0.01, format="%.3f", key="h",
                    help="h é a dimensão no plano de flexão (a direção das cargas laterais).")
else:
    sb.number_input("D (m)", min_value=0.05, step=0.01, format="%.3f", key="D")
sb.number_input("L – comprimento do pilar (m)", min_value=0.5, step=0.1, format="%.2f", key="L")

sb.subheader("3. Armadura longitudinal")
c1, c2, c3 = sb.columns(3)
c1.selectbox("Bitola (mm)", OPCOES_BITOLA, key="bitola")
c2.number_input("Cobrim. (mm)", min_value=10.0, max_value=80.0, step=5.0, format="%.0f", key="cobrimento",
                help="Cobrimento nominal da armadura de estribo.")
c3.selectbox("Estribo (mm)", OPCOES_ESTRIBO, key="estribo")
sb.checkbox("Informar d′ manualmente", key="dLinha_manual")
if val("dLinha_manual"):
    sb.number_input("d′ – distância do CG da armadura à face (m)", min_value=0.01, step=0.001, format="%.4f",
                    key="dLinha")
else:
    ss["dLinha"] = dLinha_atual()
    sb.caption(f"d′ = cobrimento + estribo + φ/2 = {val('cobrimento'):g} + {val('estribo'):g} + "
               f"{val('bitola') / 2:g} = **{dLinha_atual() * 1000:.1f} mm**")

if ss["tipo"] == "Retangular":
    sb.radio("Como definir as barras", OPCOES_ARM, key="arm_modo")
    if ss["arm_modo"] == OPCOES_ARM[0]:
        sb.caption("Cada face começa com as barras dos cantos (4 no total). Acrescente barras extras em cada face; "
                   "o desenho acima mostra o resultado (plano de flexão na horizontal).")
        c1, c2 = sb.columns(2)
        c1.number_input("Extras – faces esq./dir.", min_value=0, max_value=10, step=1, key="n_face_h",
                        help="Barras entre os cantos, em CADA uma das faces perpendiculares ao plano de flexão "
                             "(esquerda e direita no desenho).")
        c2.number_input("Extras – faces sup./inf.", min_value=0, max_value=N_MAX_CAMADAS - 2, step=1, key="n_face_b",
                        help="Barras entre os cantos, em CADA uma das faces paralelas ao plano de flexão "
                             "(superior e inferior no desenho). Cada barra extra cria uma camada intermediária.")
    else:
        sb.number_input("Número de camadas de armadura", min_value=2, max_value=N_MAX_CAMADAS, step=1, key="n_camadas",
                        help="Camadas igualmente espaçadas entre as coordenadas −h/2+d′ e h/2−d′.")
        cols = sb.columns(min(int(ss["n_camadas"]), 4))
        for i in range(int(ss["n_camadas"])):
            cols[i % len(cols)].number_input(f"Barras – camada {i + 1}", min_value=1, max_value=50, step=1,
                                             key=f"n_{i}")
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

sb.subheader("6. Impacto de veículo")
sb.radio("Ação de impacto", MODOS_IMPACTO, key="imp_modo",
         help="Escolha a categoria da NBR 6120 ou use 'Livre' para definir a força e a altura.")
if ss["imp_modo"] == MODOS_IMPACTO[1]:
    sb.selectbox("Categoria da garagem", list(mod.CATEGORIAS_GARAGEM), key="imp_cat",
                 format_func=lambda k: f"{k} – {mod.CATEGORIAS_GARAGEM[k]['desc']}")
elif ss["imp_modo"] == MODOS_IMPACTO[2]:
    sb.selectbox("Situação do pilar", list(mod.CASOS_VIA), key="imp_caso",
                 format_func=lambda k: mod.CASOS_VIA[k]["desc"])
    sb.number_input("Distância do pilar à via (m)", min_value=0.0, max_value=mod.DIST_ZERO_VIA, step=0.5,
                    format="%.1f", key="imp_dist",
                    help="As forças da Tabela 14 decrescem linearmente com a distância, sendo zero a 10 m.")
if ss["imp_modo"] in (MODOS_IMPACTO[1], MODOS_IMPACTO[2]):
    sb.radio("Direção da força", OPCOES_DIR, key="imp_dir",
             help="Fx e Fy não atuam ao mesmo tempo. Para analisar a outra direção, gire a seção (troque b e h).")
    sb.radio("Sentido", OPCOES_SENT, key="imp_sent", horizontal=True)
    if ss["imp_modo"] == MODOS_IMPACTO[1]:
        sb.checkbox("Pilar próximo a descida de rampa (forças × 2)", key="imp_rampa")
elif ss["imp_modo"] == MODOS_IMPACTO[3]:
    sb.number_input("F (kN), (+) para a direita", step=5.0, format="%.2f", key="imp_F_livre")
    sb.number_input("Altura de aplicação H (m, a partir da base)", min_value=0.0, step=0.1, format="%.2f",
                    key="imp_H_livre", help="Aplicada no nó mais próximo da malha de ΔZ. Use múltiplos de 0,1 m.")
_imp = impacto_atual()
if ss["imp_modo"] != MODOS_IMPACTO[0]:
    sb.info(_imp["resumo"])
if ss["imp_modo"] == MODOS_IMPACTO[2]:
    sb.caption("Barreira com mais de 50 cm de altura: ela resiste às forças e o pilar não precisa ser verificado.")

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
# Pré-visualização do modelo (redesenhada a cada alteração dos dados)
# --------------------------------------------------------------------------------------
def dados_desenho(imp):
    """Reúne o que os desenhos precisam; devolve (dict, avisos) ou (None, [erro]) se a geometria é inviável."""
    retangular = ss["tipo"] == "Retangular"
    d = dLinha_atual()
    bit = float(val("bitola"))
    mt = mod.propriedades_materiais(float(val("fck")), float(val("fyk")), float(val("Es")))
    L = float(val("L"))

    if retangular:
        b, h, D = float(val("b")), float(val("h")), 0.0
        if d >= min(b, h) / 2:
            return None, ["d′ precisa ser menor que metade de b e de h para desenhar a armadura."]
        n_bar = lista_n_barras()
        u, v = mod.posicoes_barras("retangular", d, n_barras=n_bar, b=b, h=h)
    else:
        b, h, D = 0.0, 0.0, float(val("D"))
        if d >= D / 2:
            return None, ["d′ precisa ser menor que D/2 para desenhar a armadura."]
        n_bar = None
        u, v = mod.posicoes_barras("circular", d, D=D, n_circ=int(val("n_barras_circ")))

    biapoiado = ss["apoio"] == OPCOES_APOIO[0]
    base = dict(
        tipo="retangular" if retangular else "circular", apoio="biapoiado" if biapoiado else "engastado",
        b=b, h=h, D=D, L=L, dLinha=d, bitola=bit, cobrimento=float(val("cobrimento")), estribo=float(val("estribo")),
        u=u, v=v, mat=mt,
        Nd=float(val("Nd")), qd=float(val("qd")), MTd=float(val("MTd")),
        MBd=float(val("MBd")) if biapoiado else 0.0, QTd=0.0 if biapoiado else float(val("QTd")),
        F=imp["F"], H=imp["H"], F_label=imp["F_label"],
    )
    avisos = mod.verificar_armadura(base["tipo"], u, v, bit, b=b, h=h, D=D, n_barras=n_bar,
                                    Nd=base["Nd"], fyd=mt["fyd"])
    base["As_cm2"] = len(u) * mod.area_barra(bit) * 1e4
    base["rho"] = len(u) * mod.area_barra(bit) / ((b * h) if retangular else np.pi * D**2 / 4)
    return base, avisos


with area_modelo:
    with st.expander("Pré-visualização do modelo", expanded=True):
        desenho, avisos_modelo = dados_desenho(_imp)
        if desenho is None:
            st.error(avisos_modelo[0])
        else:
            col_a, col_b = st.columns([1, 1.15])
            col_a.pyplot(plots.plot_secao_transversal(desenho))
            col_b.pyplot(plots.plot_modelo_estrutural(desenho))
            st.caption(f"**As = {desenho['As_cm2']:.2f} cm²** ({len(desenho['u'])}Ø{val('bitola'):g} mm) · "
                       f"taxa geométrica ρ = {desenho['rho'] * 100:.2f} % · "
                       "O sentido de cada carga no desenho vem do sinal informado; os rótulos mostram o módulo. "
                       f"Os materiais seguem o critério do código (γc = {core.GAMMA_C:g}, γs = {core.GAMMA_S:g}).")
            for a in avisos_modelo:
                st.warning(a)


# --------------------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------------------
def coletar_parametros():
    retangular = ss["tipo"] == "Retangular"
    imp = impacto_atual()
    p = dict(
        tipo_secao="retangular" if retangular else "circular",
        apoio="biapoiado" if ss["apoio"] == OPCOES_APOIO[0] else "engastado",
        L=float(val("L")), DeltaZ=float(val("DeltaZ")), dLinha=float(dLinha_atual()), bitola=float(val("bitola")),
        fck=float(val("fck")), fyk=float(val("fyk")), Es=float(val("Es")),
        Nd=float(val("Nd")), qd=float(val("qd")), MBd=float(val("MBd")), MTd=float(val("MTd")),
        QTd=float(val("QTd")), Fveic=float(imp["F"]),
        Pos_Fveic=float(imp["H"]) if float(imp["F"]) != 0 else None,
    )
    if retangular:
        p.update(b=float(val("b")), h=float(val("h")), n_barras=lista_n_barras())
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
    st.info("Confira o modelo desenhado acima, ajuste os dados na barra lateral e clique em **Calcular**.")
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
