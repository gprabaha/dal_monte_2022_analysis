"""MAT-file loader with consistent options across the project."""

import scipy.io as sio


def load_mat_from_path(path):
    """Load a MATLAB .mat file with squeezing and struct simplification."""
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
