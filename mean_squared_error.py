import torch


class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """
        computes the mean squared error between x1 (inputs) and x2 (targets)

        Arguments
        -------
        ctx: a pytorch context object
        x1: (Tensor of size (T x n) where T is the batch size and n is the number of input features.
        x2: (Tensor) of size (T x n)

        Returns
        ------
        y: (scalar) The mean squared error between x1 and x2
        """
        ctx.save_for_backward(x1, x2)
        y = torch.mean((x1 - x2)**2)
        return y

    @staticmethod
    def backward(ctx, dzdy):
        """
        back-propagates the error with respect to the input arguments

        Arguments
        --------
        ctx: A PyTorch context object
        dzdy:  a scalar (Tensor), the gradient with respect to y

        Returns
        ------
        dzdx1 (Tensor): of size(T x n), the gradients w.r.t x1
        dzdx2 (Tensor): of size(T x n), the gradients w.r.t x2
        """
        x1, x2 = ctx.saved_tensors

        m = x1.numel() # number of attributes
        dydx1 = torch.multiply(2/m, x1-x2)
        dydx2 = -torch.multiply(2/m, x1-x2)

        dzdx1 = torch.multiply(dzdy, dydx1)
        dzdx2 = torch.multiply(dzdy, dydx2)

        return dzdx1, dzdx2
