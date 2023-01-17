"""
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
      param.data = torch.from_numpy(p).to(device=self.device)
