import collections
import os
import re
import sys

root = sys.argv[1]
allowed = {'x_path', 'y_path', 'delta', 'lambda', 'do_interacts', 'y_factor', 'pure_homo',
           'thresh_fdr', 'spec', 'fdr', 'SLIDE_iter', 'slide_selection', 'interaction_size_cap',
           'nlambda', 'base_seed', 'knockoff_backend', 'rscript', 'cv'}
bad = collections.defaultdict(list)
n = 0
for dp, dn, fn in os.walk(root):
    dn[:] = [d for d in dn if d not in ('target', '.git', 'node_modules', '.pixi', 'worktrees')]
    for f in fn:
        if not f.endswith(('.yaml', '.yml')):
            continue
        p = os.path.join(dp, f)
        try:
            s = open(p).read()
        except OSError:
            continue
        if len(sys.argv) > 2 and sys.argv[2] not in p.lower():
            continue
        if "x_path" not in s:
            continue
        n += 1
        keys = {m.group(1) for m in re.finditer(r'^([A-Za-z_]\w*)\s*:', s, re.M)}
        for k in keys - allowed:
            bad[k].append(p)
print('configs with x_path:', n)
for k, v in sorted(bad.items()):
    print(k, len(v), v[:2])
