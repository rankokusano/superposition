"""Build docs/research_overview.html from docs/research_overview.md.

The Markdown file is the source of truth; the HTML is generated only (never edited by hand).
Supports the subset of Markdown used in that file: headings, paragraphs, nested lists,
pipe tables, standalone images (rendered as figures), block quotes, horizontal rules,
fenced code, and inline code / bold / italic / links. Image paths are kept as written
(relative to docs/), so the HTML must stay next to the Markdown file.

Usage: python analyze/build_research_overview_html.py [--md docs/research_overview.md] [--out docs/research_overview.html] [--embed]

--embed writes every image into the page as a data URI, so the single HTML file can be copied
and opened anywhere (default output for --embed: docs/research_overview_standalone.html).
"""
import argparse
import base64
import mimetypes
import datetime
import html
import pathlib
import re

_HERE = pathlib.Path(__file__).resolve().parent.parent  # my_research/preference_inference

CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--card:#f6f7f9;--accent:#1f5f9f}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;--fg:#e8e8ea;--mut:#a0a3a8;--line:#33363c;--card:#1f2228;--accent:#7fb2e5}}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e8e8ea;--mut:#a0a3a8;--line:#33363c;--card:#1f2228;--accent:#7fb2e5}
body{background:var(--bg);color:var(--fg);font-family:"Hiragino Sans","Noto Sans JP",sans-serif;line-height:1.75;margin:0 auto;max-width:1150px;padding:16px}
h1{font-size:1.55rem;line-height:1.4}h2{border-bottom:2px solid var(--line);padding-bottom:4px;margin-top:2.4em}
h3{margin-top:1.9em;border-left:5px solid var(--accent);padding-left:8px}h4{margin-top:1.6em}
a{color:var(--accent)}
figure{margin:14px 0 24px;background:var(--card);padding:10px;border-radius:6px}
figure img{max-width:100%;height:auto;display:block;background:#fff;margin:0 auto}
figcaption{font-size:.9rem;margin-top:6px}figcaption code{color:var(--mut);font-size:.78rem;word-break:break-all}
table{border-collapse:collapse;font-size:.86rem;margin:8px 0}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}
th{background:var(--card)}.scroll{overflow-x:auto}
td code,li code,p code{font-size:.82rem;word-break:break-all}
blockquote{margin:8px 0;padding:6px 12px;border-left:4px solid var(--mut);background:var(--card)}
nav.toc{background:var(--card);border-radius:6px;padding:10px 16px;margin:16px 0}
nav.toc ul{margin:2px 0;padding-left:1.3em}nav.toc>ul{padding-left:.6em}
.gen{font-size:.8rem;color:var(--mut)}
.top{position:fixed;right:14px;bottom:14px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 10px;font-size:.85rem;text-decoration:none}
@media print{:root{--bg:#fff;--fg:#000;--mut:#555;--line:#bbb;--card:#f4f4f4;--accent:#1f5f9f}
body{max-width:none;padding:0;font-size:10pt;line-height:1.6}.top,.gen{display:none}
h2{break-before:page}h1+nav.toc{break-after:page}h2,h3,h4{break-after:avoid}
figure{break-inside:avoid}figure img{max-height:22cm;width:auto}tr{break-inside:avoid}
table{font-size:8pt}a{color:inherit;text-decoration:none}figcaption code{display:none}}
@page{size:A4;margin:15mm 12mm}
"""

_LIST_RE = re.compile(r'^(\s*)([-*]|\d+\.)\s+(.*)$')
_IMG_LINE_RE = re.compile(r'^\s*!\[([^\]]*)\]\(([^)\s]+)\)\s*$')
_HEAD_RE = re.compile(r'^(#{1,6})\s+(.*?)\s*$')


def inline(s):
    out = []
    for part in re.split(r'(`[^`]*`)', s):
        if len(part) >= 2 and part[0] == '`' and part[-1] == '`':
            out.append('<code>' + html.escape(part[1:-1]) + '</code>')
            continue
        t = html.escape(part, quote=False)
        t = re.sub(r'\[([^\]]+)\]\(([^)\s]+)\)', r'<a href="\2">\1</a>', t)
        t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<i>\1</i>', t)
        out.append(t)
    return ''.join(out)


def plain(s):
    s = re.sub(r'`([^`]*)`', r'\1', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    return s


def split_row(line):
    cells = line.strip()
    if cells.startswith('|'):
        cells = cells[1:]
    if cells.endswith('|'):
        cells = cells[:-1]
    return [c.strip() for c in cells.split('|')]


def render_table(lines):
    head = split_row(lines[0])
    body = [split_row(l) for l in lines[2:]]
    out = ['<div class="scroll"><table>', '<tr>' + ''.join('<th>%s</th>' % inline(c) for c in head) + '</tr>']
    for row in body:
        out.append('<tr>' + ''.join('<td>%s</td>' % inline(c) for c in row) + '</tr>')
    out.append('</table></div>')
    return '\n'.join(out)


def render_list(entries):
    out, stack = [], []
    for ind, ordered, text in entries:
        tag = 'ol' if ordered else 'ul'
        if not stack or ind > stack[-1][0]:
            out.append('<%s>' % tag)
            stack.append((ind, tag))
        else:
            while len(stack) > 1 and ind < stack[-1][0]:
                out.append('</li></%s>' % stack.pop()[1])
            out.append('</li>')
        out.append('<li>' + inline(text))
    while stack:
        out.append('</li></%s>' % stack.pop()[1])
    return ''.join(out)


def render_figure(alt, src):
    cap = inline(alt)
    return ('<figure><a href="{s}" target="_blank"><img src="{s}" alt="{a}"></a>'
            '<figcaption>{c}<br><code>{p}</code></figcaption></figure>').format(
                s=html.escape(src, quote=True), a=html.escape(plain(alt), quote=True), c=cap, p=html.escape(src))


def convert(md):
    lines = md.split('\n')
    blocks, headings, used_ids = [], [], set()
    title = 'research overview'
    i, n = 0, len(lines)

    def is_block_start(l):
        return (not l.strip() or l.startswith('```') or _HEAD_RE.match(l) or l.strip() == '---'
                or l.lstrip().startswith('|') or _IMG_LINE_RE.match(l) or _LIST_RE.match(l) or l.startswith('>'))

    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith('```'):
            j = i + 1
            while j < n and not lines[j].startswith('```'):
                j += 1
            blocks.append('<pre><code>%s</code></pre>' % html.escape('\n'.join(lines[i + 1:j])))
            i = j + 1
            continue
        m = _HEAD_RE.match(line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            hid = re.sub(r'\s+', '-', plain(text)).strip('-')
            base, k = hid, 2
            while hid in used_ids:
                hid = '%s-%d' % (base, k)
                k += 1
            used_ids.add(hid)
            if level == 1:
                title = plain(text)
            else:
                headings.append((level, hid, text))
            blocks.append('<h{l} id="{i}">{t}</h{l}>'.format(l=level, i=html.escape(hid, quote=True), t=inline(text)))
            if level == 1:
                blocks.append('<!--TOC-->')
            i += 1
            continue
        if line.strip() == '---':
            blocks.append('<hr>')
            i += 1
            continue
        if line.lstrip().startswith('|'):
            j = i
            while j < n and lines[j].lstrip().startswith('|'):
                j += 1
            blocks.append(render_table(lines[i:j]))
            i = j
            continue
        m = _IMG_LINE_RE.match(line)
        if m:
            blocks.append(render_figure(m.group(1), m.group(2)))
            i += 1
            continue
        if _LIST_RE.match(line):
            entries = []
            while i < n:
                lm = _LIST_RE.match(lines[i])
                if lm:
                    entries.append([len(lm.group(1)), lm.group(2)[0].isdigit(), lm.group(3)])
                elif lines[i].strip() and lines[i].startswith(' ') and entries:
                    entries[-1][2] += ' ' + lines[i].strip()
                else:
                    break
                i += 1
            blocks.append(render_list(entries))
            continue
        if line.startswith('>'):
            j = i
            while j < n and lines[j].startswith('>'):
                j += 1
            blocks.append('<blockquote>%s</blockquote>' % inline(' '.join(l.lstrip('>').strip() for l in lines[i:j])))
            i = j
            continue
        j = i
        para = []
        while j < n and (j == i or not is_block_start(lines[j])):
            para.append(lines[j].strip())
            j += 1
        blocks.append('<p>%s</p>' % inline(' '.join(para)))
        i = j

    toc = ['<nav class="toc" id="目次"><b>目次</b>']
    depth = 1
    for level, hid, text in headings:
        if level > 4:
            continue
        while depth < level:
            toc.append('<ul>')
            depth += 1
        while depth > level:
            toc.append('</ul>')
            depth -= 1
        toc.append('<li><a href="#%s">%s</a></li>' % (html.escape(hid, quote=True), inline(text)))
    while depth > 1:
        toc.append('</ul>')
        depth -= 1
    toc.append('</nav>')
    body = '\n'.join(blocks).replace('<!--TOC-->', '\n'.join(toc), 1)
    return title, body


def embed_images(page, base_dir):
    def to_data_uri(m):
        src = html.unescape(m.group(1))
        path = (base_dir / src).resolve()
        mime = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        return '<img src="data:%s;base64,%s"' % (mime, data)
    page = re.sub(r'<a href="([^"]+)" target="_blank">(<img )', r'\2', page)
    page = re.sub(r'(<img [^>]*>)</a>', r'\1', page)
    return re.sub(r'<img src="([^"]+)"', to_data_uri, page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', default=str(_HERE / 'docs' / 'research_overview.md'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--embed', action='store_true')
    args = ap.parse_args()
    md_path = pathlib.Path(args.md)
    title, body = convert(md_path.read_text(encoding='utf-8'))
    gen = '<p class="gen">このページは <code>%s</code> から <code>analyze/build_research_overview_html.py</code> で生成したもの（%s）。手で編集しないこと。</p>' % (
        md_path.name, datetime.date.today().isoformat())
    page = ('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>修士研究の全体のまとめ</title><style>%s</style></head><body>\n%s\n%s\n'
            '<a class="top" href="#目次">目次へ</a></body></html>\n') % (CSS, gen, body)
    if args.embed:
        page = embed_images(page, md_path.resolve().parent)
    out = args.out or str(_HERE / 'docs' / ('research_overview_standalone.html' if args.embed else 'research_overview.html'))
    pathlib.Path(out).write_text(page, encoding='utf-8')
    print('wrote', out, '(title: %s)' % title)


if __name__ == '__main__':
    main()
