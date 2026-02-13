"""MAT-file loader with consistent options across the project."""

import scipy.io as sio


def load_mat_from_path(path):
    """Load a MATLAB .mat file with squeezing and struct simplification.

    Args:
        path: Path to the .mat file.

    Returns:
        Dictionary of MATLAB variables with squeezed arrays and simple structs.
    """
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
