import scipy.io as sio


def load_mat_from_path(path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
