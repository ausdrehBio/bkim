import torch


def get_device(force_cpu=False, verbose=True):
    if verbose:
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    if torch.cuda.is_available() or not force_cpu:
        if verbose:
            cuda_id = torch.cuda.current_device()
            print(f"CUDA version: {torch.version.cuda}")
            print(f"ID of current CUDA device: {torch.cuda.current_device()}")
            print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")

        return torch.device('cuda')
    else:
        return torch.device('cpu')


