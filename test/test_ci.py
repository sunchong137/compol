import numpy as np
import sys
sys.path.append("../")
import civecs

def test_gen_strs():
    norb = 4
    nelec = 2
    cistrs = civecs.gen_cistr(norb, nelec)
    ref = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [0,1,1,0],
        [1,0,0,1],
        [0,1,0,1],
        [0,0,1,1]
    ])
    print(cistrs)
    assert np.allclose(cistrs, ref)

test_gen_strs()
