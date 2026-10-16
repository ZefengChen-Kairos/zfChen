import pickle
import numpy as np

OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
      ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if (mod, name) in OK:
            return super().find_class(mod, name)
        raise pickle.UnpicklingError(f"blocked {mod}.{name}")


def load_c2ws(b):
    import io
    return [np.asarray(x, float) for x in _U(io.BytesIO(b)).load()]
