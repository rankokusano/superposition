"""
Build the v15 split field models from sim_v10's field.obj / field.mtl
(docs/v15_experiment_log.md Sec.3): the floor + ground plane as one model,
and each of the 4 corner cubes as its own model in every material it can
take (the project colour at that corner, or gray), so a cube can turn gray
within an episode by toggling visibility. Geometry, normals and the
materials of the original are copied unchanged; only face indices are
renumbered per file and a `gray` material is added.

    python v15core/make_models.py   # writes v15core/resource/model/
"""

import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'sim_v10', 'simulation', 'resource', 'model')
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resource', 'model')

# field.obj object -> arena corner (arena (x, y) = world (x, z))
CUBE_CORNER = {'Cube.001': 'TL', 'Cube.002': 'BL', 'Cube': 'BR', 'Cube.003': 'TR'}
# project colour -> field.mtl material (Camera.read_pixel reads GL_BGR)
COLOR_MTL = {'Red': 'blue', 'Green': 'green', 'Blue': 'red', 'Cyan': 'yellow', 'Gray': 'gray'}
CORNER_COLORS = {'TL': ('Red', 'Green', 'Gray'), 'BL': ('Red', 'Green', 'Gray'),
                 'BR': ('Blue', 'Gray'), 'TR': ('Cyan', 'Gray')}
GRAY_MTL = """
newmtl gray
Ns 96.078431
Ka 1.000000 1.000000 1.000000
Kd 0.500000 0.500000 0.500000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.000000
d 1.000000
illum 2
"""


def parse_objects(path):
    """-> list of (name, v lines, vt lines, vn lines, other lines, offsets)."""
    objs, cur = [], None
    nv = nvt = nvn = 0
    for line in open(path).read().splitlines():
        if line.startswith('o '):
            cur = {'name': line[2:], 'v': [], 'vt': [], 'vn': [], 'body': [],
                   'off': (nv, nvt, nvn)}
            objs.append(cur)
        elif cur is None:
            continue
        elif line.startswith('v '):
            cur['v'].append(line); nv += 1
        elif line.startswith('vt '):
            cur['vt'].append(line); nvt += 1
        elif line.startswith('vn '):
            cur['vn'].append(line); nvn += 1
        else:
            cur['body'].append(line)
    return objs


def renumber(face, off):
    out = []
    for xyz in face.split(' ')[1:]:
        parts = xyz.split('/')
        parts = [str(int(p) - off[i]) if p else p for i, p in enumerate(parts)]
        out.append('/'.join(parts))
    return 'f ' + ' '.join(out)


def write_obj(name, objs, mtl_override=None):
    lines = ['# v15 split of sim_v10 field.obj (v15core/make_models.py)', f'mtllib {name}.mtl']
    nv = nvt = nvn = 0
    for o in objs:
        off = (o['off'][0] - nv, o['off'][1] - nvt, o['off'][2] - nvn)
        lines.append('o ' + o['name'])
        lines += o['v'] + o['vt'] + o['vn']
        for b in o['body']:
            if b.startswith('f '):
                lines.append(renumber(b, off))
            elif b.startswith('usemtl ') and mtl_override:
                lines.append('usemtl ' + mtl_override)
            else:
                lines.append(b)
        nv += len(o['v']); nvt += len(o['vt']); nvn += len(o['vn'])
    with open(os.path.join(DST, name + '.obj'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(DST, name + '.mtl'), 'w') as f:
        f.write(open(os.path.join(SRC, 'field.mtl')).read() + GRAY_MTL)


def main():
    os.makedirs(DST, exist_ok=True)
    objs = {o['name']: o for o in parse_objects(os.path.join(SRC, 'field.obj'))}
    write_obj('field_base', [objs['Ground_Plane.000'], objs['Floor_Plane.014']])
    for oname, corner in CUBE_CORNER.items():
        for color in CORNER_COLORS[corner]:
            write_obj(f'cube_{corner}_{color}', [objs[oname]], COLOR_MTL[color])
    print('written to', DST)


if __name__ == '__main__':
    main()
