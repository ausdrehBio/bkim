import torch


def get_device(force_cpu=False, verbose=True):
    """
    Get the device to use for training.

    :param force_cpu: if True, force the use of the CPU
    :param verbose: if True, print information about the device
    :return: torch.device
    """
    if verbose:
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    if not force_cpu and torch.cuda.is_available():
        if verbose:
            cuda_id = torch.cuda.current_device()
            print(f"CUDA version: {torch.version.cuda}")
            print(f"ID of current CUDA device: {torch.cuda.current_device()}")
            print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

        return torch.device('cuda')
    else:
        return torch.device('cpu')
