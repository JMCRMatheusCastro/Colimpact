"""
ColImpact (Column Impact Analysis Tool) - núcleo de cálculo

Programador: José Matheus de Castro Rodrigues  |  Recife / PE, Brasil.

Este módulo contém TODA a parte numérica do notebook original (seções 2, 4, 5, 6 e 7),
sem nenhuma dependência de Streamlit ou de Matplotlib. Assim ele pode ser testado,
importado em outros scripts ou voltar para um notebook sem alterações.

Unidades internas (as mesmas do notebook): kN, m, kN/m², kN.m.
A interface (app.py) recebe fck/fyk em MPa e Es em GPa e converte aqui dentro.
"""

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------------------
# CONSTANTES (fixas no notebook original)
# --------------------------------------------------------------------------------------
GAMMA_C = 1.2        # coeficiente de ponderação do concreto
GAMMA_S = 1.0        # coeficiente de ponderação do aço
GAMMA_F3 = 1.1       # coef. p/ análise de 2ª ordem (NBR 6118, item 15.3.1)

# Deformações limites do concreto
EPS_0 = 0.002        # 2,0 ‰
EPS_U = 0.0035       # 3,5 ‰
K_EPS = 1 - (EPS_0 / EPS_U)

# Bitolas comerciais disponíveis (mm) e respectivas áreas (m²)
BITOLAS = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 32.0])
AREAS = np.array([0.0000785, 0.000122, 0.000201, 0.000314, 0.000491, 0.000804])

PASSO_LAMELA = 0.01  # (m) discretização da altura da seção para integração das tensões


def discretizar_altura(h, centrada):
    """Lamelas de ~1 cm ao longo da altura h.

    centrada=True  -> coordenadas de -h/2 a h/2 (diagrama N-M-1/r e largura da seção circular)
    centrada=False -> coordenadas de 0 a h (determinação da linha neutra)

    Usa a mesma quantidade de pontos nos dois casos, o que garante que o vetor de larguras
    b(y) da seção circular sempre tenha o mesmo tamanho que o vetor de lamelas.
    """
    n = max(int(round(h / PASSO_LAMELA)), 1)
    if centrada:
        return np.linspace(-h / 2, h / 2, n + 1)
    return np.linspace(0, h, n + 1)


# ======================================================================================
# 2. FUNÇÕES DE PROCESSAMENTO
# ======================================================================================

# --------------------------------------------------------------------------------------
# 2.1 Deslocamento lateral - Pilares engastados
# --------------------------------------------------------------------------------------
def DeslocamentoLateralENGASTE(nNos, DeltaZ, Nk, q, QT, MT, EIef, Pos_Fveic=None, Fveic=0):

    A = lil_matrix((nNos + 1, nNos + 1))
    B = np.zeros((nNos + 1, 1))

    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)

    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 1
        No_impacto_matriz = No_impacto_fisico - 2

    for i in range(nNos + 1):

        # Condição de contorno: W(0)=0
        if i == 0:
            A[i, 0] = (EIef[0]) + (-2 * Nk * (DeltaZ**2) + EIef[0] + 4 * EIef[1] + EIef[2])
            A[i, 1] = Nk * (DeltaZ**2) - 2 * EIef[1] - 2 * EIef[2]
            A[i, 2] = EIef[2]
            B[i] = q * DeltaZ**4

        # Condição de contorno: dW(0)/dz=0
        elif i == 1:
            A[i, 0] = Nk * DeltaZ**2 - 2 * EIef[1] - 2 * EIef[2]
            A[i, 1] = -2 * Nk * DeltaZ**2 + EIef[1] + 4 * EIef[2] + EIef[3]
            A[i, 2] = Nk * DeltaZ**2 - 2 * EIef[2] - 2 * EIef[3]
            A[i, 3] = EIef[3]
            B[i] = q * DeltaZ**4

        # Condição de contorno: momento fletor aplicado no topo
        elif i == nNos - 1:
            A[i, i + 1] = 0
            A[i, i] = -EIef[-2]
            A[i, i - 1] = 2 * EIef[-2]
            A[i, i - 2] = -EIef[-2]
            A[i, i - 3] = 0
            B[i] = MT * DeltaZ**2

        # Condição de contorno: força transversal aplicada no topo
        elif i == nNos:
            A[i, i] = -EIef[-1]
            A[i, i - 1] = -Nk * (DeltaZ**2) + 2 * EIef[-1]
            A[i, i - 2] = -EIef[-1] + EIef[-3]
            A[i, i - 3] = Nk * (DeltaZ**2) - 2 * EIef[-3]
            A[i, i - 4] = EIef[-3]
            B[i] = 2 * QT * DeltaZ**3

        # Linhas intermediárias
        else:
            A[i, i - 2] = EIef[i]
            A[i, i - 1] = Nk * (DeltaZ**2) - 2 * EIef[i] - 2 * EIef[i + 1]
            A[i, i] = -2 * Nk * (DeltaZ**2) + EIef[i] + 4 * EIef[i + 1] + EIef[i + 2]
            A[i, i + 1] = Nk * (DeltaZ**2) - 2 * EIef[i + 1] - 2 * EIef[i + 2]
            A[i, i + 2] = EIef[i + 2]

            if tem_impacto and i == No_impacto_matriz:
                B[i] = q * (DeltaZ**4) + Fveic * DeltaZ**3  # DeltaZ^3 porque δ ≈ 1/ΔZ
            else:
                B[i] = q * (DeltaZ**4)

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[:-2]

    W = np.zeros((nNos, 1))
    W[1:nNos, 0] = Resultado

    return W


# --------------------------------------------------------------------------------------
# 2.2 Deslocamento lateral - Pilares biapoiados
# --------------------------------------------------------------------------------------
def DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nk, q, MB, MT, EIef, Pos_Fveic=None, Fveic=0):

    A = lil_matrix((nNos + 2, nNos + 2))
    B = np.zeros((nNos + 2, 1))

    tem_impacto = (Pos_Fveic is not None) and (Fveic != 0)

    if tem_impacto:
        No_impacto_fisico = int(round(Pos_Fveic / DeltaZ)) + 2
        No_impacto_matriz = No_impacto_fisico - 1

    for i in range(nNos + 2):

        # Momento fletor aplicado na base
        if i == 0:
            A[i, 1] = -EIef[1]
            A[i, 2] = -EIef[1]
            B[i] = MB * DeltaZ**2

        # W(0)=0
        elif i == 1:
            A[i, 0] = EIef[0]
            A[i, 1] = Nk * (DeltaZ**2) - 2 * EIef[0] - 2 * EIef[1]
            A[i, 2] = Nk * (DeltaZ**2) - 2 * EIef[1] - 2 * EIef[2]
            A[i, 3] = EIef[2]
            B[i] = q * DeltaZ**4

        elif i == 2:
            A[i, 1] = EIef[1]
            A[i, 2] = -2 * Nk * (DeltaZ**2) + EIef[1] + 4 * EIef[2] + EIef[3]
            A[i, 3] = Nk * (DeltaZ**2) - 2 * EIef[2] - 2 * EIef[3]
            A[i, 4] = EIef[3]
            B[i] = q * DeltaZ**4

        elif i == 3:
            A[i, 2] = Nk * (DeltaZ**2) - 2 * EIef[2] - 2 * EIef[3]
            A[i, 3] = -2 * Nk * (DeltaZ**2) + EIef[2] + 4 * EIef[3] + EIef[4]
            A[i, 4] = Nk * (DeltaZ**2) - 2 * EIef[3] - 2 * EIef[4]
            A[i, 5] = EIef[4]
            B[i] = q * DeltaZ**4

        # W(L)=0
        elif i == nNos - 2:
            A[i, i + 1] = Nk * (DeltaZ**2) - 2 * EIef[-4] - 2 * EIef[-3]
            A[i, i] = -2 * Nk * (DeltaZ**2) + EIef[-5] + 4 * EIef[-4] + EIef[-3]
            A[i, i - 1] = Nk * (DeltaZ**2) - 2 * EIef[-5] - 2 * EIef[-4]
            A[i, i - 2] = EIef[-5]
            B[i] = q * DeltaZ**4

        elif i == nNos - 1:
            A[i, i + 1] = EIef[-2]
            A[i, i] = -2 * Nk * (DeltaZ**2) + EIef[-4] + 4 * EIef[-3] + EIef[-2]
            A[i, i - 1] = Nk * (DeltaZ**2) - 2 * EIef[-4] - 2 * EIef[-3]
            A[i, i - 2] = EIef[-4]
            B[i] = q * DeltaZ**4

        elif i == nNos:
            A[i, i + 1] = EIef[-1]
            A[i, i] = Nk * (DeltaZ**2) - 2 * EIef[-2] - 2 * EIef[-1]
            A[i, i - 1] = Nk * (DeltaZ**2) - 2 * EIef[-3] - 2 * EIef[-2]
            A[i, i - 2] = EIef[-3]
            B[i] = q * DeltaZ**4

        # Momento fletor aplicado no topo
        elif i == nNos + 1:
            A[i, i - 1] = -EIef[-2]
            A[i, i - 2] = -EIef[-2]
            B[i] = MT * DeltaZ**2

        # Linhas intermediárias
        else:
            A[i, i - 2] = EIef[i - 1]
            A[i, i - 1] = Nk * (DeltaZ**2) - 2 * EIef[i - 1] - 2 * EIef[i]
            A[i, i] = -2 * Nk * (DeltaZ**2) + EIef[i - 1] + 4 * EIef[i] + EIef[i + 1]
            A[i, i + 1] = Nk * (DeltaZ**2) - 2 * EIef[i] - 2 * EIef[i + 1]
            A[i, i + 2] = EIef[i + 1]

            if tem_impacto and i == No_impacto_matriz:
                B[i] = q * DeltaZ**4 + Fveic * DeltaZ**3  # DeltaZ^3 porque δ ≈ 1/ΔZ
            else:
                B[i] = q * DeltaZ**4

    A = A.tocsr()
    Resultado = spsolve(A, B)
    Resultado = Resultado[2:-2]
    Resultado = Resultado.reshape(-1, 1)

    W = np.vstack([0, Resultado, 0])

    return W


# --------------------------------------------------------------------------------------
# 2.3 / 2.4 Momentos fletores
# --------------------------------------------------------------------------------------
def MomentoFletorENGASTE(nNos, DeltaZ, EIef, MT, W):

    M = np.zeros(nNos)

    for i in range(1, nNos - 1):
        M[i] = -EIef[i] * (W[i - 1, 0] - 2 * W[i, 0] + W[i + 1, 0]) / DeltaZ**2

    M[0] = -EIef[0] * (2 * W[1, 0]) / DeltaZ**2
    M[-1] = MT

    return M


def MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MB, MT, W):

    M = np.zeros(nNos)

    for i in range(1, nNos - 1):
        M[i] = -EIef[i] * (W[i - 1, 0] - 2 * W[i, 0] + W[i + 1, 0]) / DeltaZ**2

    M[0] = MB
    M[-1] = MT

    return M


# --------------------------------------------------------------------------------------
# 2.4 Rigidez à flexão constante
# --------------------------------------------------------------------------------------
def RigidezFlexaoConstante(b, h, r, fck, coef):

    if r != 0:
        I = (np.pi * r**4) / 4                        # seção circular (m^4)
    else:
        I = (b * h**3) / 12                           # seção retangular (m^4)

    Ei = 5600 * (fck / 1000) ** 0.5                   # módulo tangente (MPa)
    Ecs = (0.8 + 0.2 * ((fck / 1000) / 80)) * Ei      # módulo secante (MPa)
    EIef = coef * I * Ecs * 1000                      # (kN*m²)

    return EIef


# --------------------------------------------------------------------------------------
# 2.5 Rigidez à flexão secante - Diagrama N, M, 1/r
# --------------------------------------------------------------------------------------
def rigidezsecante(χ_values, MF_values, MF, Mr):

    interp_chi = interp1d(MF_values, χ_values, kind='quadratic', fill_value="extrapolate")

    MF_limited = np.minimum(MF, MF_values[-1])

    χ_interp = interp_chi(MF_limited)
    χ_fiss = interp_chi(Mr)
    EI_const = Mr / χ_fiss

    with np.errstate(divide='ignore', invalid='ignore'):   # 0/0 onde MF=0 é descartado pelo np.where
        EI_values = np.where(MF > Mr, MF / χ_interp, EI_const)

    return EI_values


# --------------------------------------------------------------------------------------
# 2.6 / 2.7 Deformações
# --------------------------------------------------------------------------------------
def ε(χ, d, εm):
    return εm + χ * d


def εlim(d, x, h, ε0, εu, k, ds):
    xlim = (εu / (εu + 0.01)) * ds[0]

    if isinstance(d, np.ndarray):
        epsilon = np.full_like(d, np.nan, dtype=float)
    else:
        epsilon = np.nan

    x_scalar = x.item() if isinstance(x, np.ndarray) and x.size == 1 else x

    if np.isnan(x_scalar):
        return epsilon

    if 0 <= x_scalar < xlim:
        denominator = ds[0] - x_scalar
        if np.isclose(denominator, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = 0.01 * ((x_scalar - d) / denominator)
    elif xlim <= x_scalar <= h:
        if np.isclose(x_scalar, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = εu * ((x_scalar - d) / x_scalar)
    elif x_scalar > h:
        denominator = x_scalar - k * h
        if np.isclose(denominator, 0.0):
            epsilon = np.inf * np.sign(x_scalar - d)
        else:
            epsilon = ε0 * ((x_scalar - d) / denominator)

    return epsilon


# --------------------------------------------------------------------------------------
# 2.8 Tensão no concreto
# --------------------------------------------------------------------------------------
def σconcrete(εc, ε0, εu, fcd, coef):
    """Função de tensão do concreto - versão vetorizada"""
    σc = np.zeros_like(εc)

    fck_MPa = (fcd * GAMMA_C) / 1000

    if fck_MPa <= 40:
        ηc = 1.0
    else:
        ηc = (40 / fck_MPa) ** (1 / 3)

    if fck_MPa <= 50:
        n = 2.0
    else:
        n = 1.4 + 23.4 * ((90 - fck_MPa) / 100) ** 4

    mask_parabolic = (εc > 0) & (εc <= ε0)
    mask_rectangular = (εc > ε0) & (εc <= εu)

    if coef == 1.1:
        σc[mask_parabolic] = 1.1 * fcd * (1 - (1 - εc[mask_parabolic] / ε0) ** n)
        σc[mask_rectangular] = 1.1 * fcd
    else:
        σc[mask_parabolic] = coef * ηc * fcd * (1 - (1 - εc[mask_parabolic] / ε0) ** n)
        σc[mask_rectangular] = coef * ηc * fcd

    return σc


# --------------------------------------------------------------------------------------
# 2.9 Tensão no aço (elastoplástico perfeito) - vetorizada
# --------------------------------------------------------------------------------------
def σsteel(εs, fyd, Es):
    # |ε| <= εyd -> Es*ε ; caso contrário sign(ε)*fyd  (equivale a limitar Es*ε em ±fyd)
    return np.clip(Es * np.asarray(εs, dtype=float), -fyd, fyd)


# --------------------------------------------------------------------------------------
# 2.10 Determinação da linha neutra
# --------------------------------------------------------------------------------------
def EquiDasForcasEMomentos(x, b, h, ds, n, fcd, fyd, Es, ε0, εu, k, N, M):

    dc = discretizar_altura(h, centrada=False)
    εc = εlim(dc, x, h, ε0, εu, k, ds)
    εs = εlim(ds, x, h, ε0, εu, k, ds)
    σc = σconcrete(εc, ε0, εu, fcd, 0.85)
    σs = σsteel(εs, fyd, Es)
    Rcc = integrate.trapezoid(σc * b, dc)
    Mc = integrate.trapezoid(σc * dc * b, dc)

    EQ = (N - Rcc) * np.sum(n * σs * ds) + (M - N * (h / 2) + Mc) * np.sum(n * σs)

    return EQ


# --------------------------------------------------------------------------------------
# 2.11 Diagrama N, M, 1/r
# --------------------------------------------------------------------------------------
def EquiDasForcas(b, h, dc, ds, As, fcd, fyd, Es, ε0, εu, N, εm, χ, coef):

    εc = ε(χ, dc, εm)
    εs = ε(χ, ds, εm)
    σc = σconcrete(εc, ε0, εu, fcd, coef)
    σs = σsteel(εs, fyd, Es)
    Rcc = integrate.trapezoid(σc * b, dc)
    Rs = np.sum(As * σs)
    EQ = N - Rcc - Rs

    return EQ


def MomentoEquilibrante(b, h, dc, ds, As, fcd, fyd, Es, ε0, εu, N, εm, χ, coef):

    εc = ε(χ, dc, εm)
    εs = ε(χ, ds, εm)
    σc = σconcrete(εc, ε0, εu, fcd, coef)
    σs = σsteel(εs, fyd, Es)
    Mc = integrate.trapezoid(σc * dc * b, dc)
    Ms = np.sum(As * σs * ds)
    MF = Mc + Ms

    return MF


def DiagramaNMr(b, h, ds, As, fcd, fyd, Es, ε0, εu, N, coef, Mdbreak=None,
                incremento_χ=0.00001, max_passos=100000):

    dc = discretizar_altura(h, centrada=True)

    χ_values = []
    M_values = []
    εm_chute = 0.001
    χ = 0.0

    for _ in range(max_passos):
        εm_raiz = fsolve(lambda εm: EquiDasForcas(b, h, dc, ds, As, fcd, fyd, Es, ε0, εu, N, εm, χ, coef), εm_chute)

        M = MomentoEquilibrante(b, h, dc, ds, As, fcd, fyd, Es, ε0, εu, N, εm_raiz, χ, coef)
        M = float(np.asarray(M).ravel()[0])

        deformacoes = ε(χ, dc, εm_raiz[0])

        if Mdbreak is None:
            # Curva 0.85: para quando alguma fibra atinge os limites de deformação
            if np.any(deformacoes > 0.0035) or np.any(deformacoes < -0.01):
                break
        else:
            # Curva 1.1: para quando o momento ultrapassa Mdbreak
            if M > Mdbreak:
                break

        χ_values.append(χ)
        M_values.append(M)

        εm_chute = εm_raiz[0]
        χ += incremento_χ
    else:
        raise RuntimeError(
            "O diagrama N-M-1/r não atingiu o critério de parada. "
            "Verifique a armadura, o esforço normal e as dimensões da seção."
        )

    return np.array(χ_values), np.array(M_values)


# --------------------------------------------------------------------------------------
# 2.12 Momento de fissuração
# --------------------------------------------------------------------------------------
def momento_fissuracao(fck, b, h, D, x_raiz, alpha=1.5):

    fct_m = 0.3 * (fck / 1000) ** (2 / 3) * 1000      # kN/m²
    fct = 0.7 * fct_m                                 # kN/m²

    if D != 0:
        Ic = (np.pi * (D / 2) ** 4) / 4               # seção circular (m^4)
        yt = D - x_raiz
    else:
        Ic = (b * h**3) / 12                          # seção retangular (m^4)
        yt = h - x_raiz

    Mr = alpha * fct * (Ic / yt)                      # kN.m

    return Mr


# --------------------------------------------------------------------------------------
# 2.13 Armadura longitudinal pré-definida
# --------------------------------------------------------------------------------------
def ArmaduraPredefinida(M_NLG, Nd, b, h, ds, n, bitola, fcd, fyd, Es, ε0, εu, k):

    Md = max(np.absolute(M_NLG))
    x_raiz, _, ier, msg = fsolve(
        lambda x: EquiDasForcasEMomentos(x, b, h, ds, n, fcd, fyd, Es, ε0, εu, k, Nd, Md),
        h / 2, full_output=True)
    As = n * AREAS[BITOLAS == bitola][0]

    return Md, x_raiz, As, (ier == 1), msg


# --------------------------------------------------------------------------------------
# 2.14 Deslocamentos e momentos fletores (NLG)
# --------------------------------------------------------------------------------------
def AnaliseNLG(apoio, nNos, DeltaZ, b, h, D, fck, Nd, qd, MTd, QTd, MBd, Pos_Fveic, Fveic):

    if apoio == 'engastado':
        nEI = nNos + 1
    elif apoio == 'biapoiado':
        nEI = nNos + 2
    else:
        raise ValueError("apoio deve ser 'engastado' ou 'biapoiado'")

    if D != 0:
        EIef = RigidezFlexaoConstante(0, 0, D / 2, fck, 1) * np.ones(nEI)
    else:
        EIef = RigidezFlexaoConstante(b, h, 0, fck, 1) * np.ones(nEI)

    if apoio == 'engastado':
        W_P0 = DeslocamentoLateralENGASTE(nNos, DeltaZ, 0, qd, QTd, MTd, EIef, Pos_Fveic, Fveic)
        M_P0 = MomentoFletorENGASTE(nNos, DeltaZ, EIef, MTd, W_P0)

        W_NLG = DeslocamentoLateralENGASTE(nNos, DeltaZ, Nd, qd, QTd, MTd, EIef, Pos_Fveic, Fveic)
        M_NLG = MomentoFletorENGASTE(nNos, DeltaZ, EIef, MTd, W_NLG)

    else:
        W_P0 = DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, 0, qd, MBd, MTd, EIef, Pos_Fveic, Fveic)
        M_P0 = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MBd, MTd, W_P0)

        W_NLG = DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nd, qd, MBd, MTd, EIef, Pos_Fveic, Fveic)
        M_NLG = MomentoFletorBIAPOIADO(nNos, DeltaZ, EIef, MBd, MTd, W_NLG)

    return EIef, W_P0, M_P0, W_NLG, M_NLG


# --------------------------------------------------------------------------------------
# 2.15 Deslocamentos e momentos fletores (NLG e NLF)
# --------------------------------------------------------------------------------------
def AnaliseNLG_NLF(apoio, nNos, DeltaZ, b, h, D, fck, Nd, qd, MTd, QTd, MBd, Pos_Fveic, Fveic,
                   EIef, x_raiz, χ_values, M_values, γf3, tol=1e-4, max_iter=100):
    """Igual à versão do notebook, mas em vez de imprimir devolve o histórico de resíduos.

    Retorna: EIef_NL_final, W_NLG_NLF, M_NLG_NLF, historico (lista de resíduos), convergiu (bool)
    """

    # Os esforços locais de 2ª ordem, conforme o item 15.3.1 da ABNT NBR 6118, podem ser calculados
    # com as cargas majoradas por γf / γf3, que posteriormente são majoradas por γf3 = 1.1.

    if apoio == 'engastado':
        deslocamento = lambda EI: DeslocamentoLateralENGASTE(nNos, DeltaZ, Nd / γf3, qd / γf3, QTd / γf3, MTd / γf3, EI, Pos_Fveic, Fveic)
        momento = lambda EI, W: MomentoFletorENGASTE(nNos, DeltaZ, EI, MTd / γf3, W)
        pad = (0, 1)
    elif apoio == 'biapoiado':
        deslocamento = lambda EI: DeslocamentoLateralBIAPOIADOS(nNos, DeltaZ, Nd / γf3, qd / γf3, MBd / γf3, MTd / γf3, EI, Pos_Fveic, Fveic)
        momento = lambda EI, W: MomentoFletorBIAPOIADO(nNos, DeltaZ, EI, MBd / γf3, MTd / γf3, W)
        pad = (1, 1)
    else:
        raise ValueError("apoio deve ser 'engastado' ou 'biapoiado'")

    W_prev = deslocamento(EIef)
    M_prev = momento(EIef, W_prev)

    Mr = np.asarray(momento_fissuracao(fck, b, h, D, x_raiz)).item()

    EIef_NL = rigidezsecante(χ_values, M_values, np.abs(M_prev), Mr)
    EIef_NL_corrigido = np.pad(EIef_NL, pad, mode='edge')

    EIef_NL_final = np.copy(EIef_NL)
    iter_count = 0
    historico = []
    convergiu = False

    while iter_count < max_iter:
        iter_count += 1

        W_NLG_NLF = deslocamento(EIef_NL_corrigido)
        M_NLG_NLF = momento(EIef_NL_final, W_NLG_NLF)

        EIef_NL_final = rigidezsecante(χ_values, M_values, np.absolute(M_NLG_NLF), Mr)

        norma_W = np.linalg.norm(W_NLG_NLF)
        diff_W = np.linalg.norm(W_NLG_NLF - W_prev) / norma_W if norma_W > 0 else 0.0
        historico.append(float(diff_W))

        if diff_W < tol:
            convergiu = True
            break

        W_prev = np.copy(W_NLG_NLF)
        EIef_NL_corrigido = np.pad(EIef_NL_final, pad, mode='edge')

    return EIef_NL_final, W_NLG_NLF, M_NLG_NLF, historico, convergiu, Mr


# ======================================================================================
# 3. PÓS-PROCESSAMENTO (tabelas)
# ======================================================================================
def tabela_deslocamentos(datatable):
    linhas = []
    for item in datatable:
        Z = np.asarray(item['Z']).flatten()
        linhas.append({'Curva': item['label'],
                       'Desl. Máx. (m)': float(np.max(Z)),
                       'Desl. Mín. (m)': float(np.min(Z))})
    return pd.DataFrame(linhas)


def tabela_momentos_fletores(datatable):
    linhas = []
    for item in datatable:
        Z = np.asarray(item['M']).flatten()
        linhas.append({'Curva': item['label'],
                       'MF Máx. (kN.m)': float(np.max(Z)),
                       'MF Mín. (kN.m)': float(np.min(Z))})
    return pd.DataFrame(linhas)


def tabela_fs(arr_momentos, m_resistente):
    """Entrada: vetor Base (0) -> Topo (L). Saída: tabela com Topo na primeira linha e Base na última.
    O menor F.S. é destacado com ⚠️."""

    fs_numericos = []
    for m in arr_momentos:
        if abs(m) < 0.001:
            fs_numericos.append(float('inf'))
        else:
            fs_numericos.append(abs(m_resistente / m))

    min_fs = min(fs_numericos)

    fs_display = []
    for val in fs_numericos:
        if val == float('inf'):
            fs_display.append("(OK!)")
        else:
            val_fmt = round(val, 2)
            if val == min_fs:
                fs_display.append(f"{val_fmt:.2f} ⚠️")
            else:
                fs_display.append(f"{val_fmt:.2f}")

    num_pontos = len(arr_momentos)
    labels_z = []
    for i in range(num_pontos):
        ratio = i / (num_pontos - 1) if num_pontos > 1 else 0.0
        if ratio == 0.0:
            labels_z.append("0 (Base)")
        elif ratio >= 0.999:
            labels_z.append("L (Topo)")
        else:
            labels_z.append(f"{ratio:.2f} L")

    df = pd.DataFrame({
        'z': labels_z,
        'Msd (kN.m)': np.round(arr_momentos, 2),
        'F.S.': fs_display,
    })

    return df.iloc[::-1].reset_index(drop=True)


# ======================================================================================
# MONTAGEM DOS DADOS DE ENTRADA (seção 1 do notebook) + VALIDAÇÃO
# ======================================================================================
def montar_secao(p):
    """Converte o dicionário de entradas do usuário nas variáveis geométricas/de armadura."""
    tipo = p['tipo_secao']
    L = float(p['L'])
    DeltaZ = float(p['DeltaZ'])
    nNos = int(round(L / DeltaZ)) + 1
    dLinha = float(p['dLinha'])

    s = dict(tipo=tipo, apoio=p['apoio'], L=L, DeltaZ=DeltaZ, nNos=nNos,
             dLinha=dLinha, bitola=float(p['bitola']))

    if tipo == 'retangular':
        b, h = float(p['b']), float(p['h'])
        n = np.array(p['n_barras'], dtype=float)
        nLinha = len(n)
        ds = np.linspace(-h / 2 + dLinha, h / 2 - dLinha, nLinha)
        s.update(b=b, h=h, D=0, n=n, ds=ds)

    elif tipo == 'circular':
        D = float(p['D'])
        h = D
        discre = discretizar_altura(h, centrada=True)
        b = 2 * np.sqrt(np.clip((D / 2) ** 2 - discre**2, 0, None))
        nLinha = int(p['n_barras_circ'])
        R_efet = (D / 2) - dLinha
        Δθ = (2 * np.pi) / nLinha
        θ = np.pi / 2 + np.arange(nLinha) * Δθ
        ds = R_efet * np.sin(θ)
        ds = np.unique(np.round(ds, 6))
        n = np.ones_like(ds)
        n[1:-1] = 2
        s.update(b=b, h=h, D=D, n=n, ds=ds)
    else:
        raise ValueError("tipo_secao deve ser 'retangular' ou 'circular'")

    return s


def validar_entradas(p):
    """Devolve uma lista de mensagens de erro (vazia se estiver tudo certo)."""
    erros = []
    tipo = p['tipo_secao']
    L = float(p['L'])
    dz = float(p['DeltaZ'])
    nNos = int(round(L / dz)) + 1

    if L <= 0:
        erros.append("O comprimento L deve ser maior que zero.")
    if dz <= 0:
        erros.append("O passo de discretização ΔZ deve ser maior que zero.")
    if nNos < 10:
        erros.append(f"Discretização muito grossa ({nNos} nós). Reduza ΔZ ou aumente L (mínimo de 10 nós).")
    if abs(L / dz - round(L / dz)) > 1e-6:
        erros.append("L deve ser múltiplo de ΔZ.")

    if float(p['bitola']) not in BITOLAS:
        erros.append("Bitola não disponível.")

    if tipo == 'retangular':
        b, h = float(p['b']), float(p['h'])
        if b <= 0 or h <= 0:
            erros.append("b e h devem ser maiores que zero.")
        if len(p['n_barras']) < 2:
            erros.append("A seção retangular precisa de pelo menos 2 camadas de armadura.")
        if any(v <= 0 for v in p['n_barras']):
            erros.append("Cada camada de armadura precisa ter ao menos 1 barra.")
        if float(p['dLinha']) >= h / 2:
            erros.append("d' deve ser menor que h/2.")
    else:
        D = float(p['D'])
        nb = int(p['n_barras_circ'])
        if D <= 0:
            erros.append("O diâmetro D deve ser maior que zero.")
        if nb < 4 or nb % 2 != 0:
            erros.append("O número de barras da seção circular deve ser par e maior ou igual a 4.")
        if float(p['dLinha']) >= D / 2:
            erros.append("d' deve ser menor que D/2.")

    if float(p['Fveic']) != 0:
        pos = p.get('Pos_Fveic')
        if pos is None or not (0 < float(pos) < L):
            erros.append("A posição do impacto deve estar entre 0 e L.")

    cargas_laterais = [float(p['qd']), float(p['MTd']), float(p['Fveic']),
                       float(p['MBd']) if p['apoio'] == 'biapoiado' else float(p['QTd'])]
    if all(c == 0 for c in cargas_laterais):
        erros.append("Informe ao menos um carregamento lateral (qd, momento de extremidade, força no topo ou impacto): "
                     "só com esforço normal os deslocamentos são nulos.")

    if float(p['fck']) <= 0 or float(p['fyk']) <= 0 or float(p['Es']) <= 0:
        erros.append("fck, fyk e Es devem ser maiores que zero.")

    return erros


# ======================================================================================
# EXECUÇÃO COMPLETA (seções 4 a 8 do notebook)
# ======================================================================================
def executar_analise(p, tol=1e-4, max_iter=100):
    """Roda toda a análise e devolve um dicionário com os resultados.

    Parâmetros (dicionário `p`):
      tipo_secao: 'retangular' | 'circular'      apoio: 'biapoiado' | 'engastado'
      b, h | D, L, dLinha, bitola(mm), n_barras(lista) | n_barras_circ, DeltaZ
      fck (MPa), fyk (MPa), Es (GPa)
      Nd, qd, MBd, MTd, QTd, Fveic, Pos_Fveic
    """
    erros = validar_entradas(p)
    if erros:
        raise ValueError("\n".join(erros))

    s = montar_secao(p)
    apoio, L, DeltaZ, nNos = s['apoio'], s['L'], s['DeltaZ'], s['nNos']
    b, h, D, n, ds, bitola = s['b'], s['h'], s['D'], s['n'], s['ds'], s['bitola']

    # Materiais (kN/m²)
    fck = float(p['fck']) * 1000
    fcd = fck / GAMMA_C
    fyk = float(p['fyk']) * 1000
    fyd = fyk / GAMMA_S
    Es = float(p['Es']) * 1e6

    # Carregamentos
    Nd = float(p['Nd'])
    qd = float(p['qd'])
    MTd = float(p['MTd'])
    MBd = float(p['MBd']) if apoio == 'biapoiado' else 0.0     # não se aplica a pilar engastado
    QTd = float(p['QTd']) if apoio == 'engastado' else 0.0     # não se aplica a pilar biapoiado
    Fveic = float(p['Fveic'])
    Pos_Fveic = float(p['Pos_Fveic']) if (Fveic != 0 and p.get('Pos_Fveic') is not None) else None

    avisos = []

    # 4. Deslocamentos e momentos (NLG)
    EIef, W_P0, M_P0, W_NLG, M_NLG = AnaliseNLG(
        apoio, nNos, DeltaZ, b, h, D, fck, Nd, qd, MTd, QTd, MBd, Pos_Fveic, Fveic)

    # 5. Armadura longitudinal
    Md, x_raiz, As, ok_x, msg_x = ArmaduraPredefinida(
        M_NLG, Nd, b, h, ds, n, bitola, fcd, fyd, Es, EPS_0, EPS_U, K_EPS)
    if not ok_x:
        avisos.append("A determinação da linha neutra (fsolve) não convergiu: " + str(msg_x))

    # 6. Diagrama N, M, 1/r
    χ85, M85 = DiagramaNMr(b, h, ds, As, fcd, fyd, Es, EPS_0, EPS_U, Nd, coef=0.85)
    Mdbreak = M85[-1] / 1.1
    χ11, M11 = DiagramaNMr(b, h, ds, As, fcd, fyd, Es, EPS_0, EPS_U, Nd / 1.1, coef=1.1, Mdbreak=Mdbreak)

    # 7. Deslocamentos e momentos (NLG e NLF)
    EIef_NL, W_NL, M_NL, historico, convergiu, Mr = AnaliseNLG_NLF(
        apoio, nNos, DeltaZ, b, h, D, fck, Nd, qd, MTd, QTd, MBd, Pos_Fveic, Fveic,
        EIef, x_raiz, χ11, M11, GAMMA_F3, tol=tol, max_iter=max_iter)
    if not convergiu:
        avisos.append("Número máximo de iterações atingido sem convergência da rigidez não linear.")

    # 8. Resultados derivados
    Mrd = float(M85[-1])                       # momento fletor resistente (kN.m)
    M_NL_maj = GAMMA_F3 * M_NL                 # momentos de cálculo NLG & NLF (kN.m)
    z = np.linspace(0, L, nNos)
    with np.errstate(divide='ignore', invalid='ignore'):
        fsi = np.abs(Mrd / M_NL_maj)
    i_min = int(np.argmin(fsi))

    return dict(
        z=z, L=L, nNos=nNos, apoio=apoio, tipo=s['tipo'],
        EIef=EIef, EIef_NL=EIef_NL,
        W_P0=W_P0, M_P0=M_P0, W_NLG=W_NLG, M_NLG=M_NLG, W_NLG_NLF=W_NL, M_NLG_NLF=M_NL,
        M_NLG_NLF_maj=M_NL_maj,
        chi85=χ85, M85=M85, chi11=χ11, M11=M11,
        Mrd=Mrd, Md=float(Md), Mr=float(Mr), x_raiz=float(np.asarray(x_raiz).ravel()[0]),
        As_camadas=np.asarray(As), As_total_cm2=float(np.sum(As) * 1e4),
        ds=ds, n_camadas=n,
        fsi=fsi, fsi_min=float(fsi[i_min]), z_critico=float(z[i_min]),
        historico=historico, convergiu=convergiu,
        avisos=avisos,
        gamma_f3=GAMMA_F3,
    )
