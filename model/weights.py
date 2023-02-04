
from torch import nn
import torch.optim as optim
import torch
import numpy as np


def get_weights(self):
    with torch.no_grad():
      w = []
      for name, param in self.model.named_parameters():
        w.append(param.data.clone().detach().cpu().numpy())
      return w

#braucht das das NN-model???? jp mb
def set_weights(self, w):
    """
    Parameters
    ----------
    w : numpy.array
      networks weights with arbitrary dimensions
    """
    with torch.no_grad():
        for i, (name, param) in enumerate(self.model.named_parameters()):
            p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
            param.data = torch.from_numpy(p).to(device=torch.device)

      
# code update von mb aus mohammad repo fc.deepl
def average_weights(self, params, client_model):
    global_weights = [np.array(client_model.get_weights(), dtype='object') * 0] * self.load('n_splits')
    total_n_samples = [0] * self.load('n_splits')
    for client_models in params:
        for model_counter, (weights, n_samples) in enumerate(client_models):
            global_weights[model_counter] += np.array(weights, dtype='object') * n_samples
            total_n_samples[model_counter] += n_samples
        updated_weights = []
        for counter, (w, n) in enumerate(zip(global_weights, total_n_samples)):
            updated_weights.append(w / n)
        return updated_weights


def get_avg_params(params):
    """
    Get the average of the parameters

    :param models: List of models to get parameters from
    :return: List of average parameters
    """
    # Align the parameters and average them
    params_avg = [sum(p) / len(p) for p in zip(*params)]
    return params_avg


def get_avg_model_params(models):
    """
    Get the average of the parameters of the models

    :param models: List of models to get parameters from
    :return: List of average parameters
    """
    # Get the parameters of the models
    params = [model.get_parameters() for model in models]
    # Align the parameters and average them
    return get_avg_params(params)


def set_model_params(model, params):
    """
    Set the parameters of the model to the average of the parameters
    :param model: Model to set parameters for
    :param params: Parameters to set
    :return: None
    """
    params = map(lambda p: torch.from_numpy(p), params)
    with torch.no_grad():
        for model_params, p in zip(model.parameters(), params):
            model_params.data.copy_(p)
