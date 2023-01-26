import torch


class FullyConnected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        """
        Computes the output of the fully_connected function given in the assignment

        Arguments
        ---------
        ctx: a PyTorch context object
        x (Tensor): of size (T x n), the input features
        w (Tensor): of size (n x m), the weights
        b (Tensor): of size (m), the biases

        Returns
        -----
        y (Tensor): of size (T x m), the outputs of the fully_connected operator
        """
        ctx.save_for_backward(x, w, b)
        y = torch.mm(x, w) + b

        return y

    @staticmethod
    def backward(ctx, dz_dy):
        """
        back-propagates the gradients with respect to the inputs
        ctx: a PyTorch context object.
        dz_dy (Tensor): of size (T x m), the gradients with respect to the output argument y

        Returns
        -------
        dzdx (Tensor): of size (T x n), the gradients with respect to x
        dzdw (Tenor): of size (n x m), the gradients with respect to w
        dzdb (Tensor): of size (m), the gradients with respect to b
        """
        x, w, b = ctx.saved_tensors
        dydx = w
        dydw = x
        dydb = 1

        dzdx = torch.mm(dz_dy, dydx.t())
        dzdw = torch.mm(dz_dy.t(), dydw).t()
        dzdb = torch.sum(dz_dy, 0)

        return dzdx, dzdw, dzdb     # retrieved by x.grad, w.grad, b.grad
