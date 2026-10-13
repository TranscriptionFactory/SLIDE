"""Drop hunks containing a given added line from a unified diff read on stdin."""
import sys

needle = sys.argv[1]
out, hunk, header = [], [], []
in_hunk = False


def flush():
    if hunk and not any(line.startswith('+') and needle in line for line in hunk):
        out.extend(hunk)


for line in sys.stdin.read().splitlines(keepends=True):
    if line.startswith('diff --git'):
        flush()
        hunk = []
        in_hunk = False
        out.append(line)
    elif line.startswith('@@'):
        flush()
        hunk = [line]
        in_hunk = True
    elif in_hunk:
        hunk.append(line)
    else:
        out.append(line)
flush()
sys.stdout.write(''.join(out))
