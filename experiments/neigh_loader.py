import numpy as np


def is_number(src_str):
    try:
        float(src_str)
        return True
    except ValueError:
        return False


def file2array(_path, _max_lines=None):
    _f = open(_path, 'r')

    _state = 0
    _res = []
    _buff = []

    _res = [[]]
    for _l in _f.readlines():
        _segs = _l.split()
        if is_number(_segs[0]):
            _buff = []
            for _seg in _segs:
                _buff.append(float(_seg))
            _res[-1].append(_buff)
        else:
            _res.append([])
    _res.pop(-1)
    return np.array(_res)
