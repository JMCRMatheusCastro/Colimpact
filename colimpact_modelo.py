"""
ColImpact - definição do modelo (sem Streamlit e sem Matplotlib)

Reúne tudo o que a interface precisa para MONTAR e CONFERIR o modelo antes de calcular:
  - ações de impacto de veículos da ABNT NBR 6120:2019 (Tabelas 13 e 14);
  - armadura por faces (cantos + barras extras), d' a partir de cobrimento e estribo;
  - coordenadas das barras (para o desenho da seção) e verificações simples da armadura.

Unidades: kN, m (como o núcleo). Bitola, cobrimento e estribo entram em mm.
"""

import numpy as np

import colimpact_core as core

# --------------------------------------------------------------------------------------
# NBR 6120:2019 - impacto acidental de veículos
# --------------------------------------------------------------------------------------
# Fx: força paralela ao fluxo dos veículos | Fy: perpendicular ao fluxo | H: altura de aplicação (m)
# As forças são ações excepcionais (NBR 8681) e atuam de forma NÃO concomitante (Fx ou Fy).
CATEGORIAS_GARAGEM = {
    "I":   dict(desc="PBT ≤ 30 kN · altura máx. 2,3 m",     Fx=100.0, Fy=50.0,  H=0.5),
    "II":  dict(desc="PBT ≤ 90 kN · altura máx. 2,6 m",     Fx=180.0, Fy=90.0,  H=0.5),
    "III": dict(desc="PBT ≤ 160 kN · altura máx. 3,0 m",    Fx=240.0, Fy=120.0, H=1.0),
    "IV":  dict(desc="PBT > 160 kN · altura máx. > 3,0 m",  Fx=320.0, Fy=160.0, H=1.0),
    "V":   dict(desc="PBT ≤ 230 kN · altura máx. ≥ 4,5 m",  Fx=320.0, Fy=160.0, H=1.0),
}

CASOS_VIA = {
    "sem_protecao": dict(desc="Pilar sem proteção",                      Fx=200.0, Fy=100.0, H=1.0),
    "guia_15":      dict(desc="Proteção de guia com h ≤ 15 cm",          Fx=150.0, Fy=75.0,  H=1.0),
    "guia_50":      dict(desc="Proteção de guia com 15 cm < h ≤ 50 cm",  Fx=100.0, Fy=50.0,  H=1.0),
}

DIST_ZERO_VIA = 10.0    # m - as forças da Tabela 14 decrescem linearmente até zero a 10 m da via
FATOR_RAMPA = 2.0       # pilares próximos a descidas de rampas: dobro das colunas 6 e 7 da Tabela 13


def fator_distancia_via(distancia):
    """Fator de redução linear da Tabela 14 (1,0 junto à via; 0,0 a 10 m ou mais)."""
    return float(min(1.0, max(0.0, 1.0 - float(distancia) / DIST_ZERO_VIA)))


def impacto_nbr6120(fonte, chave, direcao="y", sentido=1, rampa=False, distancia=0.0):
    """Força de impacto e altura de aplicação segundo a NBR 6120:2019.

    fonte     'garagem' (Tabela 13) | 'via' (Tabela 14)
    chave     categoria ('I'...'V') | caso da via ('sem_protecao', 'guia_15', 'guia_50')
    direcao   'x' (paralela ao fluxo) | 'y' (perpendicular ao fluxo)
    sentido   +1 (para a direita) | -1 (para a esquerda)
    rampa     só Tabela 13: dobra as forças
    distancia só Tabela 14: distância do pilar à via (m)
    """
    if fonte == "garagem":
        d = CATEGORIAS_GARAGEM[chave]
        fator = FATOR_RAMPA if rampa else 1.0
    elif fonte == "via":
        d = CASOS_VIA[chave]
        fator = fator_distancia_via(distancia)
    else:
        raise ValueError("fonte deve ser 'garagem' ou 'via'")
    base = d["Fx"] if direcao == "x" else d["Fy"]
    F = base * fator
    return dict(F=float(sentido) * F, F_abs=F, F_tabela=base, fator=fator, H=d["H"], direcao=direcao)


# --------------------------------------------------------------------------------------
# Armadura
# --------------------------------------------------------------------------------------
def dLinha_de_cobrimento(cobrimento_mm, estribo_mm, bitola_mm):
    """d' (m) = cobrimento + estribo + metade da bitola longitudinal."""
    return (float(cobrimento_mm) + float(estribo_mm) + float(bitola_mm) / 2.0) / 1000.0


def camadas_por_faces(n_extra_h, n_extra_b):
    """Barras por camada de uma seção retangular definida por faces.

    n_extra_h: barras extras (além dos cantos) em CADA face perpendicular ao plano de flexão (±h/2)
    n_extra_b: barras extras em CADA face paralela ao plano de flexão -> geram camadas intermediárias
               (cada uma com 2 barras, uma em cada face)
    A armadura resulta simétrica, que é o caso em que o diagrama N-M-1/r do ColImpact vale nos dois sentidos.
    """
    n_extra_h, n_extra_b = int(n_extra_h), int(n_extra_b)
    return [2 + n_extra_h] + [2] * n_extra_b + [2 + n_extra_h]


def posicoes_barras(tipo, dLinha, n_barras=None, b=None, h=None, D=None, n_circ=None):
    """Coordenadas (u, v) do centro de cada barra, em m, com origem no centro da seção.

    u: eixo do plano de flexão (o mesmo do desenho do pilar; as camadas do núcleo ficam em u = ds)
    v: eixo perpendicular (largura b)
    """
    if tipo == "retangular":
        n_barras = [int(x) for x in n_barras]
        ds = np.linspace(-h / 2 + dLinha, h / 2 - dLinha, len(n_barras))
        u, v = [], []
        for ds_k, n_k in zip(ds, n_barras):
            vs = np.linspace(-b / 2 + dLinha, b / 2 - dLinha, n_k) if n_k > 1 else np.array([0.0])
            u += [ds_k] * n_k
            v += list(vs)
        return np.array(u), np.array(v)
    # circular: mesma construção do núcleo (primeira barra em θ = 90°)
    R = D / 2 - dLinha
    theta = np.pi / 2 + np.arange(int(n_circ)) * (2 * np.pi / int(n_circ))
    return R * np.sin(theta), R * np.cos(theta)


def area_barra(bitola_mm):
    """Área (m²) da bitola comercial usada pelo núcleo."""
    return float(core.AREAS[core.BITOLAS == float(bitola_mm)][0])


def propriedades_materiais(fck_mpa, fyk_mpa, Es_gpa):
    """Resistências de cálculo e módulo do concreto, no mesmo critério do núcleo (kN/m²)."""
    fck = fck_mpa * 1000.0
    fyk = fyk_mpa * 1000.0
    Ei = 5600.0 * np.sqrt(fck_mpa)                              # MPa
    Ecs = (0.8 + 0.2 * (fck_mpa / 80.0)) * Ei * 1000.0          # kN/m²
    return dict(fck=fck, fcd=fck / core.GAMMA_C, fyk=fyk, fyd=fyk / core.GAMMA_S,
                Es=Es_gpa * 1e6, Ecs=float(Ecs))


def verificar_armadura(tipo, u, v, bitola_mm, b=None, h=None, D=None, n_barras=None, Nd=0.0, fyd=None):
    """Alertas rápidos sobre a armadura desenhada (não substituem o detalhamento)."""
    avisos = []
    phi = bitola_mm / 1000.0
    As = len(u) * area_barra(bitola_mm)
    Ac = b * h if tipo == "retangular" else np.pi * D**2 / 4

    if len(u) > 1:
        dx = u[:, None] - u[None, :]
        dy = v[:, None] - v[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        dist[np.diag_indices(len(u))] = np.inf
        livre = float(dist.min()) - phi
        minimo = max(0.02, phi)
        if livre < minimo - 1e-9:
            avisos.append(f"Espaçamento livre entre barras de {livre * 1000:.0f} mm, menor que o mínimo de "
                          f"{minimo * 1000:.0f} mm (NBR 6118, item 18.4.2.1, sem considerar o agregado graúdo).")

    if As > 0.08 * Ac:
        avisos.append(f"As = {As * 1e4:.2f} cm² excede 8 % de Ac ({0.08 * Ac * 1e4:.2f} cm²).")
    if fyd:
        As_min = max(0.15 * abs(Nd) / fyd, 0.004 * Ac)
        if As < As_min:
            avisos.append(f"As = {As * 1e4:.2f} cm² abaixo da armadura mínima de {As_min * 1e4:.2f} cm² "
                          "(0,4 % de Ac ou 0,15 Nd/fyd).")

    if tipo == "retangular" and n_barras is not None and n_barras[0] != n_barras[-1]:
        avisos.append("Armadura assimétrica: o diagrama N-M-1/r é calculado para um único sentido de curvatura, "
                      "então o FSI pode não ser confiável onde o momento muda de sinal. Prefira armadura simétrica.")
    return avisos
