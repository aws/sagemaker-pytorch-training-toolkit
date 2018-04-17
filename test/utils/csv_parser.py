import numpy as np
from six import StringIO

def loads(data):
    stream = StringIO(data)
    return np.genfromtxt(stream, dtype=np.float32, delimiter=',')


def dumps(data):
    stream = StringIO()
    np.savetxt(stream, data, delimiter=',', fmt='%s')
    return stream.getvalue()
