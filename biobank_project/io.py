"""Input/Output Utilities"""

def read_lines(file):
    with open(file) as f:
        lines = [l.strip() for l in f.readlines()]

    return lines

