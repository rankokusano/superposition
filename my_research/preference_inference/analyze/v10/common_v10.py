"""Shared helpers for v10 analysis scripts (outputs under data/result/v10_swap/)."""

import os

PI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(PI_DIR, 'data', 'result', 'v10_swap')


def git_hash():
    """HEAD commit read from .git directly (the container has no git binary)."""
    git_dir = os.path.join(PI_DIR, '..', '..', '.git')
    try:
        head = open(os.path.join(git_dir, 'HEAD')).read().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        ref_file = os.path.join(git_dir, ref)
        if os.path.exists(ref_file):
            return open(ref_file).read().strip()
        for line in open(os.path.join(git_dir, 'packed-refs')):
            if line.strip().endswith(' ' + ref):
                return line.split()[0]
    except OSError:
        pass
    return 'unknown'
