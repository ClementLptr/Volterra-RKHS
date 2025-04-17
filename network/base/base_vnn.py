import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVNN(nn.Module):
    def __init__(self):
        super(BaseVNN, self).__init__()

    def forward(self, x):
        raise NotImplementedError("The 'forward' method must be implemented in the derived class.")

    def _volterra_kernel_approximation(self, tensor1, tensor2, num_terms, num_channels_out):
        """
        Approximates the Volterra kernel by combining pairwise multiplicative interactions 
        between two input tensors.

        Args:
            tensor1 (torch.Tensor): First input tensor, shape (batch_size, channels, depth, height, width).
            tensor2 (torch.Tensor): Second input tensor, shape (batch_size, 2*num_terms*num_channels_out, depth, height, width).
            num_terms (int): Number of multiplicative terms in the kernel approximation.
            num_channels_out (int): Number of output channels for the final result.

        Returns:
            torch.Tensor: Approximated tensor, shape (batch_size, num_channels_out, depth, height, width).
        """
        tensor_mul = torch.mul(tensor2[:, 0:num_terms * num_channels_out, :, :, :], 
                                tensor2[:, num_terms * num_channels_out:2 * num_terms * num_channels_out, :, :, :])
        
        tensor_add = torch.zeros_like(tensor1)
        
        for q in range(num_terms):
            tensor_add = torch.add(tensor_add, tensor_mul[:, (q * num_channels_out):((q * num_channels_out) + num_channels_out), :, :, :])
                
        return tensor_add