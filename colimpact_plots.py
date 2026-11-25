"""
ColImpact - gráficos (Matplotlib, sem pyplot global)

Todas as funções devolvem um objeto Figure, que o Streamlit exibe com st.pyplot(fig)
e que pode ser exportado em alta resolução com fig_to_png(fig, dpi=600).
"""

import io

import matplotlib
import numpy as np
from matplotlib.figure import Figure

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
