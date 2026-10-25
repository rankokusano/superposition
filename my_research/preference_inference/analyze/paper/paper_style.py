"""
Shared style for the paper figures (paper/check/paper_round1_log.md): Japanese font, fixed condition colours,
and saving as vector PDF (TrueType embedded) plus a PNG for checking, into paper/figs/.
The container has no Japanese font; copy one from the host first (paper/figs/README.md):
  docker exec kusano_research mkdir -p /tmp/paperfonts
  docker cp /usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf kusano_research:/tmp/paperfonts/
Fonts are embedded as TrueType (Type 42). matplotlib 3.4 embeds the whole font file, so save() rewrites the PDF with
ghostscript (pdfwrite, -dSubsetFonts=true) to keep only the glyphs used. IPAPGothic (proportional IPA Gothic) is a plain TrueType file (.ttf);
the Noto CJK collection (.ttc, CFF outlines) cannot be embedded as Type 42 nor read back by ghostscript.
"""

import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
from matplotlib import font_manager  # noqa

_HERE = os.path.dirname(os.path.abspath(__file__))
PI = os.path.dirname(os.path.dirname(_HERE))
RES = os.path.join(PI, 'data', 'result')
FIGS = os.path.join(PI, 'paper', 'figs')
FONT_DIR = '/tmp/paperfonts'
FONT = 'IPAPGothic'

# condition colours (fixed per condition in every figure; markers differ as well, so colour is never the only cue)
C_NONE, C_50, C_80, C_NS = '#7a7a7a', '#eb6834', '#2a78d6', '#1baf7a'
M_NONE, M_50, M_80, M_NS = 'o', 's', '^', 'D'
C_MAIN, C_EXTRA = '0.45', '#eb6834'      # pairs 1-3 grey solid, pairs 4-6 orange dashed (central figure)
COND_LABEL = {'none': '経験なし', '50': '経験50%', '80': '経験80%', 'ns': '非共有'}


def setup():
    fonts = glob.glob(os.path.join(FONT_DIR, 'ipagp.ttf'))
    if not fonts:
        sys.exit(f'no Japanese font in {FONT_DIR}; see the module docstring')
    for f in fonts:
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({
        'font.family': FONT, 'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.0,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, f'{name}.png'), dpi=200)
    full = os.path.join(FIGS, f'{name}_fullfont.pdf')
    fig.savefig(full)
    subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                    '-dSubsetFonts=true', '-dEmbedAllFonts=true', '-dAutoRotatePages=/None',
                    f'-sOutputFile={os.path.join(FIGS, name + ".pdf")}', full], check=True)
    os.remove(full)
    print('wrote', os.path.join(FIGS, name + '.{pdf,png}'))


def pair_label(s):
    """seed index 0..5 -> 組1..組6"""
    return f'組{int(s) + 1}'
