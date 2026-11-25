"""
ColImpact - gráficos (Matplotlib, sem pyplot global)

Todas as funções devolvem um objeto Figure, que o Streamlit exibe com st.pyplot(fig)
e que pode ser exportado em alta resolução com fig_to_png(fig, dpi=600).
"""

import io

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle

RC = {
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
}

TEXTOS = {
    'pt': dict(
        curv='Curvatura ($\\chi$) [1/m]', mom='Momento Fletor [kN.m]', rig='Rigidez (EI) [kN.m²]',
        desl='Deslocamento [m]', pos='Posição ao longo do pilar [m]',
        fsi='Fator de Segurança de Impacto (FSI)', fsi_calc='FSI Calculado',
        limite='Limite ($M_{sd} = M_{rd}$)', codigo='Código Implementado',
        linear='Linear', nao_linear='Não Linear',
    ),
    'en': dict(
        curv='Curvature ($\\chi$) [1/m]', mom='Bending Moment [kN.m]', rig='Flexural Stiffness (EI) [kN.m²]',
        desl='Displacement [m]', pos='Position along the column [m]',
        fsi='Impact Safety Factor (ISF)', fsi_calc='Calculated ISF',
        limite='Limit ($M_{sd} = M_{rd}$)', codigo='Implemented Code',
        linear='Linear', nao_linear='Nonlinear',
    ),
}


def fig_to_png(fig, dpi=600):
    """Exporta a figura em PNG (bytes) para st.download_button."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def _nova_figura(dpi):
    return Figure(figsize=(9, 6), dpi=dpi, layout='tight')


# --------------------------------------------------------------------------------------
# 3.1 Diagrama N-M-1/r
# --------------------------------------------------------------------------------------
def plot_Momento_curvatura(r, language='pt', dpi=150):
    t = TEXTOS[language]
    with matplotlib.rc_context(RC):
        fig = _nova_figura(dpi)
        ax = fig.subplots()
        ax.plot(r['chi85'], r['M85'], '-', color='red', linewidth=2, label=f"0.85 fcd - {t['codigo']}")
        ax.plot(r['chi11'], r['M11'], '--', color='red', linewidth=2, label=f"1.1 fcd - {t['codigo']}")
        ax.set_xlabel(t['curv'])
        ax.set_ylabel(t['mom'])
        ax.legend(loc='lower right', frameon=True)
    return fig


# --------------------------------------------------------------------------------------
# 3.2 Momento fletor x rigidez
# --------------------------------------------------------------------------------------
def plot_Momento_vs_Rigidez(r, language='pt', dpi=150):
    t = TEXTOS[language]
    EIef_plot = r['EIef'][0] * np.ones_like(r['M_P0'])
    with matplotlib.rc_context(RC):
        fig = _nova_figura(dpi)
        ax = fig.subplots()
        ax.plot(r['M_P0'], EIef_plot, linestyle='-', color='r', label=t['linear'])
        ax.plot(r['M_NLG_NLF'], r['EIef_NL'], linestyle='--', color='b', label=t['nao_linear'])
        ax.set_xlabel(t['mom'])
        ax.set_ylabel(t['rig'])
        ax.legend(loc='best', frameon=True)
    return fig


# --------------------------------------------------------------------------------------
# 3.4 / 3.5 Deslocamentos e momentos ao longo do pilar
# --------------------------------------------------------------------------------------
_ESTILOS = ['-', '--', '-.', ':']


def _plot_ao_longo_do_pilar(series, chave, xlabel, ylabel, L, dpi):
    with matplotlib.rc_context(RC):
        fig = _nova_figura(dpi)
        ax = fig.subplots()
        ax.plot([0, 0], [0, L], color='black', linewidth=1.5, label='_nolegend_', zorder=1)   # eixo do pilar
        for i, s in enumerate(series):
            vals = np.asarray(s[chave]).flatten()
            y = np.linspace(0, L, len(vals))
            ax.plot(vals, y, linestyle=_ESTILOS[i % len(_ESTILOS)], color='red', label=s['label'],
                    linewidth=1.5, zorder=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, L)
        ax.legend(loc='best', frameon=True)
    return fig


def series_deslocamentos(r, language='pt'):
    c = TEXTOS[language]['codigo']
    return [{'Z': r['W_P0'], 'label': f"LG & LF - {c}"},
            {'Z': r['W_NLG'], 'label': f"NLG & LF - {c}"},
            {'Z': r['W_NLG_NLF'], 'label': f"NLG & NLF - {c}"}]


def series_momentos(r, language='pt'):
    c = TEXTOS[language]['codigo']
    return [{'M': r['M_P0'], 'label': f"LG & LF - {c}"},
            {'M': r['M_NLG'], 'label': f"NLG & LF - {c}"},
            {'M': r['M_NLG_NLF_maj'], 'label': f"NLG & NLF - {c}"}]


def plot_deslocamentos(r, language='pt', dpi=150):
    t = TEXTOS[language]
    return _plot_ao_longo_do_pilar(series_deslocamentos(r, language), 'Z', t['desl'], t['pos'], r['L'], dpi)


def plot_momentos_fletores(r, language='pt', dpi=150):
    t = TEXTOS[language]
    return _plot_ao_longo_do_pilar(series_momentos(r, language), 'M', t['mom'], t['pos'], r['L'], dpi)


# --------------------------------------------------------------------------------------
# 3.8 Fator de segurança de impacto
# --------------------------------------------------------------------------------------
def plot_fator_seguranca_final(r, language='pt', limite_escala=35.0, dpi=150):
    t = TEXTOS[language]
    arr_momentos = r['M_NLG_NLF_maj']
    m_resistente = r['Mrd']
    L = r['L']

    z_vals = np.linspace(0, L, len(arr_momentos))
    with np.errstate(divide='ignore', invalid='ignore'):
        fs_vals = np.abs(m_resistente / arr_momentos)
    fs_plot = np.clip(fs_vals, 0, limite_escala)

    min_fs_idx = np.argmin(fs_vals)
    min_fs = fs_vals[min_fs_idx]
    z_critico = z_vals[min_fs_idx]

    with matplotlib.rc_context(RC):
        fig = _nova_figura(dpi)
        ax = fig.subplots()

        ax.axvspan(0, 1.0, color='#ffe6e6', alpha=1.0, zorder=0)                       # zona de falha
        ax.axvline(x=1.0, color='#d62728', linestyle='--', linewidth=1.5, zorder=1, label=t['limite'])
        ax.text(1.15, L * 0.96, '$M_{sd} = M_{rd}$', color='#d62728', fontsize=10,
                ha='left', va='top', fontweight='bold')

        color_line = '#003366'
        ax.plot(fs_plot, z_vals, color=color_line, linestyle='-', linewidth=1.5, zorder=3, label=t['fsi_calc'])
        ax.fill_betweenx(z_vals, 0, fs_plot, color=color_line, alpha=0.1, zorder=2)

        # Assíntotas (infinito)
        is_inf = (fs_vals >= limite_escala)
        if np.any(is_inf):
            changes = np.diff(np.concatenate(([0], is_inf.astype(int), [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            for start, end in zip(starts, ends):
                mid_idx = (start + end) // 2
                z_inf = z_vals[mid_idx]
                ax.annotate(r'$\infty$', xy=(limite_escala, z_inf + 0.1),
                            xytext=(limite_escala - 1.5, z_inf + 0.1),
                            arrowprops=dict(arrowstyle='->', color=color_line, linewidth=1.5),
                            fontsize=16, color=color_line, va='center', ha='right', zorder=5)

        # Ponto crítico
        ax.scatter([min_fs], [z_critico], color='white', edgecolors='red', s=60, linewidth=1.5, zorder=4)
        ax.annotate(f'{min_fs:.2f}', xy=(min_fs, z_critico), xytext=(min_fs + 2.0, z_critico),
                    arrowprops=dict(arrowstyle='-|>', color='black', shrinkA=5, shrinkB=5, linewidth=1.0),
                    fontsize=12, fontweight='bold', color='red', va='center', ha='left',
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7))

        ax.set_xlabel(t['fsi'])
        ax.set_ylabel(t['pos'])
        ax.set_xlim(0, limite_escala)
        ax.set_ylim(0, L)

        ticks = np.unique(np.concatenate(([0, 1], np.arange(5, limite_escala + 1, 5))))
        ax.set_xticks(ticks)
        labels = [str(int(v)) for v in ticks]
        labels[-1] = r'$\geq$' + str(int(limite_escala))
        ax.set_xticklabels(labels)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

        ax.legend(loc='best', frameon=True)
    return fig


# ======================================================================================
# DESENHOS DO MODELO (pré-visualização ao vivo, antes de calcular)
# ======================================================================================
RC_DESENHO = {
    'font.family': 'STIXGeneral',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.grid': False,
}
COR_CONCRETO = '#c8c8c8'


def _fmt(x, casas=2):
    """Número sem zeros à direita: 1127.90 -> '1127.9', 100.00 -> '100'."""
    s = f"{x:.{casas}f}"
    return s.rstrip('0').rstrip('.') if '.' in s else s


def _seta(ax, p0, p1, lw=1.5, escala=16, cor='k', z=8):
    ax.annotate('', xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle='->', color=cor, lw=lw, mutation_scale=escala,
                                shrinkA=0, shrinkB=0))


def _marca(ax, x, y, s):
    """Traço inclinado de extremidade de cota (como no desenho de referência)."""
    ax.plot([x - s, x + s], [y - s, y + s], color='k', lw=1.0, solid_capstyle='butt', zorder=6)


def _cota_v(ax, x, y0, y1, texto, ref0, ref1, tk, fs=10):
    """Cota vertical em x, de y0 a y1. ref0/ref1: x de onde partem as linhas de chamada."""
    lado = 1.0 if x >= ref0 else -1.0
    ax.plot([ref0, x + lado * tk], [y0, y0], color='k', lw=0.6, zorder=5)
    ax.plot([ref1, x + lado * tk], [y1, y1], color='k', lw=0.6, zorder=5)
    ax.plot([x, x], [y0, y1], color='k', lw=0.8, zorder=5)
    _marca(ax, x, y0, tk)
    _marca(ax, x, y1, tk)
    ax.text(x + lado * tk * 1.6, (y0 + y1) / 2, texto, rotation=90, ha='left' if lado > 0 else 'right',
            va='center', fontsize=fs)


def _cota_h(ax, y, x0, x1, texto, ref0, ref1, tk, fs=10):
    """Cota horizontal em y, de x0 a x1. ref0/ref1: y de onde partem as linhas de chamada."""
    lado = 1.0 if y >= ref0 else -1.0
    ax.plot([x0, x0], [ref0, y + lado * tk], color='k', lw=0.6, zorder=5)
    ax.plot([x1, x1], [ref1, y + lado * tk], color='k', lw=0.6, zorder=5)
    ax.plot([x0, x1], [y, y], color='k', lw=0.8, zorder=5)
    _marca(ax, x0, y, tk)
    _marca(ax, x1, y, tk)
    ax.text((x0 + x1) / 2, y + lado * tk * 1.6, texto, ha='center',
            va='bottom' if lado > 0 else 'top', fontsize=fs)


def _arco_momento(ax, centro, raio, horario, a0, a1, rotulo, pos_rotulo, ha, fs=11):
    """Arco com seta. Anti-horário percorre a0 -> a1; horário percorre a1 -> a0 (ângulos em graus)."""
    ang = np.radians(np.linspace(a1, a0, 60) if horario else np.linspace(a0, a1, 60))
    xs = centro[0] + raio * np.cos(ang)
    ys = centro[1] + raio * np.sin(ang)
    ax.plot(xs, ys, color='k', lw=1.5, zorder=7)
    _seta(ax, (xs[-3], ys[-3]), (xs[-1], ys[-1]), lw=1.5)
    ax.text(pos_rotulo[0], pos_rotulo[1], rotulo, ha=ha, va='center', fontsize=fs)


# --------------------------------------------------------------------------------------
# Seção transversal + propriedades + carregamentos
# --------------------------------------------------------------------------------------
def plot_secao_transversal(m, dpi=120):
    """Seção vista de cima, com o plano de flexão na horizontal (o mesmo do desenho do pilar).

    m: dict com tipo, b, h, D, dLinha, bitola, cobrimento, estribo, u, v (centros das barras),
       mat (fck, fcd, fyk, fyd, Es, Ecs em kN/m²) e carregamentos (Nd, qd, MTd, MBd, QTd, F, H, F_label).
    """
    circular = m['tipo'] == 'circular'
    W = m['D'] if circular else m['h']       # extensão no plano de flexão (horizontal)
    Hh = m['D'] if circular else m['b']      # extensão perpendicular (vertical)
    sc = max(W, Hh)
    tk = 0.018 * sc
    d = m['dLinha']
    phi = m['bitola'] / 1000.0
    e_est = (m['cobrimento'] + m['estribo'] / 2) / 1000.0

    with matplotlib.rc_context(RC_DESENHO):
        fig = Figure(figsize=(5.6, 7.0), dpi=dpi, layout='tight')
        gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.9])
        ax = fig.add_subplot(gs[0])
        axt = fig.add_subplot(gs[1])

        # --- concreto e estribo
        if circular:
            ax.add_patch(Circle((0, 0), W / 2, fc=COR_CONCRETO, ec='k', lw=1.4, zorder=1))
            ax.add_patch(Circle((0, 0), W / 2 - e_est, fill=False, ec='#222222', lw=1.1, zorder=2))
        else:
            ax.add_patch(Rectangle((-W / 2, -Hh / 2), W, Hh, fc=COR_CONCRETO, ec='k', lw=1.4, zorder=1))
            ax.add_patch(Rectangle((-W / 2 + e_est, -Hh / 2 + e_est), W - 2 * e_est, Hh - 2 * e_est,
                                   fill=False, ec='#222222', lw=1.1, zorder=2))

        # --- barras longitudinais
        for uu, vv in zip(m['u'], m['v']):
            ax.add_patch(Circle((uu, vv), phi / 2, fc='k', ec='k', zorder=5))

        # --- rótulo "nØφ" com chamadas
        n_bar = len(m['u'])
        x_t, y_t = -W / 2, Hh / 2 + 0.36 * sc
        rotulo = f"{n_bar}Ø{_fmt(m['bitola'], 1)} mm"
        if circular:
            rotulo += f"   (d′ = {d * 1000:.1f} mm)"
            idx = sorted({0, n_bar // 4, n_bar // 2, (3 * n_bar) // 4})[:4]     # chamadas espalhadas no círculo
        else:
            idx = np.argsort(-(m['u']**2 + m['v']**2))[:4]                     # cantos
        ax.text(x_t - 0.05 * sc, y_t, rotulo, ha='left', va='bottom', fontsize=12)
        for i in idx:
            ax.annotate('', xy=(m['u'][i], m['v'][i]), xytext=(x_t + 0.12 * sc, y_t - 0.02 * sc),
                        arrowprops=dict(arrowstyle='->', lw=0.6, color='k', shrinkA=0, shrinkB=3), zorder=6)

        # --- cotas
        if circular:
            _cota_h(ax, -Hh / 2 - 0.16 * sc, -W / 2, W / 2, f"D = {m['D']:.2f} m", -Hh / 2, -Hh / 2, tk)
        else:
            _cota_h(ax, -Hh / 2 - 0.16 * sc, -W / 2, W / 2, f"h = {m['h']:.2f} m", -Hh / 2, -Hh / 2, tk)
            _cota_v(ax, W / 2 + 0.17 * sc, -Hh / 2, Hh / 2, f"b = {m['b']:.2f} m", W / 2, W / 2, tk)
            # d' (como no desenho de referência): horizontal no canto superior direito, vertical no inferior esquerdo
            _cota_h(ax, Hh / 2 + 0.13 * sc, W / 2 - d, W / 2, f"{d * 1000:.1f} mm",
                    Hh / 2 - d, Hh / 2, tk)
            _cota_v(ax, -W / 2 - 0.13 * sc, -Hh / 2, -Hh / 2 + d, f"{d * 1000:.1f} mm",
                    -W / 2, -W / 2 + d, tk)

        # --- direção do plano de flexão
        y_f = -Hh / 2 - 0.42 * sc
        _seta(ax, (-0.28 * sc, y_f), (0.28 * sc, y_f), lw=1.2, escala=13)
        ax.text(0, y_f - 0.04 * sc, "plano de flexão", ha='center', va='top', fontsize=10, color='#444444')

        ax.set_aspect('equal')
        ax.set_xlim(-W / 2 - 0.30 * sc, W / 2 + 0.38 * sc)
        ax.set_ylim(y_f - 0.18 * sc, Hh / 2 + 0.52 * sc)
        ax.axis('off')

        # --- painel de texto (materiais e carregamentos), como no desenho de referência
        axt.axis('off')
        mt = m['mat']
        linhas = [
            (0.97, "Propriedades do Material", 'center', 'bold', 13),
            (0.83, f"Armadura:   $f_{{yk}}$ = {_fmt(mt['fyk'])} kN/m²    $f_{{yd}}$ = {_fmt(mt['fyd'])} kN/m²", 'left', 'normal', 11),
            (0.70, f"$E_s$ = {mt['Es']:.0f} kN/m²", 'left', 'normal', 11),
            (0.57, f"Concreto:   $f_{{ck}}$ = {_fmt(mt['fck'])} kN/m²    $f_{{cd}}$ = {_fmt(mt['fcd'])} kN/m²", 'left', 'normal', 11),
            (0.44, f"$E_{{cs}}$ = {mt['Ecs']:.0f} kN/m²", 'left', 'normal', 11),
            (0.29, "Carregamentos", 'center', 'bold', 13),
        ]
        cargas = []
        if m['Nd'] != 0:
            cargas.append(f"$N_d$ = {_fmt(abs(m['Nd']), 2)} kN")
        if m['F'] != 0:
            cargas.append(f"{m['F_label']} = {_fmt(abs(m['F']))} kN (H = {_fmt(m['H'])} m)")
        if m['qd'] != 0:
            cargas.append(f"$q_d$ = {_fmt(abs(m['qd']))} kN/m")
        if m['MTd'] != 0:
            cargas.append(f"$M_{{Td}}$ = {_fmt(abs(m['MTd']))} kNm")
        if m['apoio'] == 'biapoiado' and m['MBd'] != 0:
            cargas.append(f"$M_{{Bd}}$ = {_fmt(abs(m['MBd']))} kNm")
        if m['apoio'] == 'engastado' and m['QTd'] != 0:
            cargas.append(f"$Q_{{Td}}$ = {_fmt(abs(m['QTd']))} kN")
        if not cargas:
            cargas = ["(nenhum carregamento informado)"]
        meio = (len(cargas) + 1) // 2
        for k, grupo in enumerate([cargas[:meio], cargas[meio:]]):
            if grupo:
                linhas.append((0.15 if k == 0 else 0.03, "      ".join(grupo), 'left', 'normal', 11))
        for y, txt, ha, peso, fs in linhas:
            x = 0.5 if ha == 'center' else 0.0
            axt.text(x, y, txt, ha=ha, va='center', fontsize=fs, fontweight=peso, transform=axt.transAxes)
        axt.set_xlim(0, 1)
        axt.set_ylim(0, 1)
    return fig


# --------------------------------------------------------------------------------------
# Pilar: vinculação, cargas, sentidos e cotas
# --------------------------------------------------------------------------------------
def plot_modelo_estrutural(m, dpi=120):
    """Elevação do pilar com apoios, cargas (no sentido real, pelo sinal) e cotas.

    Convenções do ColImpact: qd e F (+) para a direita; MBd (+) horário; MTd (+) anti-horário;
    QTd (+) para a direita. Os valores nos rótulos são sempre o módulo; o sentido vem do desenho.
    """
    L = m['L']
    w = 0.095 * L
    tk = 0.012 * L
    biapoiado = m['apoio'] == 'biapoiado'
    F, H = m['F'], m['H']
    # cotas do lado oposto ao da força de impacto (como no desenho de referência)
    sd = -1.0 if F >= 0 else 1.0

    with matplotlib.rc_context(RC_DESENHO):
        fig = Figure(figsize=(6.2, 7.0), dpi=dpi, layout='tight')
        ax = fig.subplots()

        # --- pilar
        ax.add_patch(Rectangle((-w / 2, 0), w, L, fc=COR_CONCRETO, ec='k', lw=1.5, zorder=3))

        # --- apoios
        if biapoiado:
            # base: apoio fixo (triângulo + chão hachurado)
            ax.add_patch(Polygon([(0, 0), (-0.04 * L, -0.06 * L), (0.04 * L, -0.06 * L)],
                                 closed=True, fc='white', ec='k', lw=1.3, zorder=4))
            ax.plot([-0.08 * L, 0.08 * L], [-0.06 * L, -0.06 * L], color='k', lw=1.3, zorder=4)
            for xh in np.linspace(-0.075 * L, 0.055 * L, 6):
                ax.plot([xh, xh + 0.025 * L], [-0.06 * L, -0.085 * L], color='k', lw=0.8, zorder=4)
            # topo: apoio móvel horizontal (triângulo + roletes), do lado das cotas
            xf = sd * w / 2
            xb = sd * (w / 2 + 0.05 * L)
            ax.add_patch(Polygon([(xf, L - 0.03 * L), (xb, L - 0.075 * L), (xb, L + 0.015 * L)],
                                 closed=True, fc='white', ec='k', lw=1.3, zorder=4))
            for yr in (L - 0.058 * L, L - 0.03 * L, L - 0.002 * L):
                ax.add_patch(Circle((sd * (w / 2 + 0.06 * L), yr), 0.0085 * L, fc='white', ec='k', lw=1.0, zorder=4))
            xp = sd * (w / 2 + 0.077 * L)
            ax.plot([xp, xp], [L - 0.08 * L, L + 0.02 * L], color='k', lw=1.3, zorder=4)
            for yh in np.linspace(L - 0.08 * L, L + 0.0 * L, 5):
                ax.plot([xp, xp + sd * 0.022 * L], [yh, yh + 0.02 * L], color='k', lw=0.8, zorder=4)
        else:
            # engaste na base
            ax.plot([-0.1 * L, 0.1 * L], [0, 0], color='k', lw=2.0, zorder=4)
            for xh in np.linspace(-0.095 * L, 0.075 * L, 8):
                ax.plot([xh, xh + 0.025 * L], [0, -0.03 * L], color='k', lw=0.8, zorder=4)

        # --- esforço normal e momento no topo
        y_nd = L + 0.19 * L
        if m['Nd'] != 0:
            if m['Nd'] > 0:
                _seta(ax, (0, y_nd), (0, L), lw=1.5)
            else:
                _seta(ax, (0, L), (0, y_nd), lw=1.5)
            ax.text(0, y_nd + 0.015 * L, f"$N_d$ = {_fmt(abs(m['Nd']))} kN", ha='center', va='bottom', fontsize=11.5)
        if m['MTd'] != 0:
            _arco_momento(ax, (0, L + 0.095 * L), 0.06 * L, horario=m['MTd'] < 0, a0=-35, a1=215,
                          rotulo=f"$M_{{Td}}$ = {_fmt(abs(m['MTd']))} kNm",
                          pos_rotulo=(-sd * 0.085 * L, L + 0.11 * L), ha='right' if sd > 0 else 'left')

        # --- momento na base (biapoiado)
        if biapoiado and m['MBd'] != 0:
            _arco_momento(ax, (0, -0.03 * L), 0.115 * L, horario=m['MBd'] > 0, a0=180, a1=360,
                          rotulo=f"$M_{{Bd}}$ = {_fmt(abs(m['MBd']))} kNm",
                          pos_rotulo=(0.0, -0.20 * L), ha='center')

        # --- força transversal no topo (engastado)
        if (not biapoiado) and m['QTd'] != 0:
            sq = 1.0 if m['QTd'] > 0 else -1.0
            _seta(ax, (0, L - 0.02 * L), (sq * (w / 2 + 0.17 * L), L - 0.02 * L), lw=1.6, escala=22)
            ax.text(sq * (w / 2 + 0.03 * L), L + 0.005 * L, f"$Q_{{Td}}$ = {_fmt(abs(m['QTd']))} kN",
                    ha='left' if sq > 0 else 'right', va='bottom', fontsize=11.5)

        # --- carga distribuída: a "caixa" fica do lado de onde a carga vem; setas apontam para o pilar
        if m['qd'] != 0:
            sqd = 1.0 if m['qd'] < 0 else -1.0      # qd < 0 (para a esquerda) -> caixa à direita
            x_near = sqd * (w / 2 + 0.24 * L)
            x_far = sqd * (w / 2 + 0.36 * L)
            ax.add_patch(Rectangle((min(x_near, x_far), 0), abs(x_far - x_near), L, fill=False, ec='k', lw=0.9, zorder=5))
            for zz in np.linspace(0, L, 17):
                _seta(ax, (x_far, zz), (x_near, zz), lw=0.9, escala=9, z=6)
            ax.text(sqd * (w / 2 + 0.38 * L), L * 0.5, f"$q_d$ = {_fmt(abs(m['qd']))} kN/m",
                    ha='left' if sqd > 0 else 'right', va='center', fontsize=11.5)

        # --- impacto de veículo
        if F != 0:
            sf = 1.0 if F > 0 else -1.0
            _seta(ax, (0, H), (sf * (w / 2 + 0.17 * L), H), lw=1.8, escala=26, z=9)
            ax.text(sf * (w / 2 + 0.095 * L), H + 0.022 * L, f"{m['F_label']}\n{_fmt(abs(F))} kN",
                    ha='center', va='bottom', fontsize=11.5, linespacing=1.1)
            _cota_v(ax, sd * (w / 2 + 0.43 * L), 0, H, f"{H:.2f} m", sd * w / 2, sd * w / 2, tk)

        # --- comprimento
        ref_topo = sd * (w / 2 + 0.077 * L) if biapoiado else sd * w / 2
        _cota_v(ax, sd * (w / 2 + 0.57 * L), 0, L, f"{L:.2f} m", sd * w / 2, ref_topo, tk)

        ax.set_aspect('equal')
        lim = w / 2 + 0.72 * L
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.27 * L, L + 0.30 * L)
        ax.axis('off')
    return fig
