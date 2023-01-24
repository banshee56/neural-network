from mean_squared_error import MeanSquaredError
import torch


def mean_squared_error_test():
    """
     Unit tests for the MeanSquaredError autograd Function.

    PROVIDED CONSTANTS
    ------------------
    TOL (float): the absolute error tolerance for the backward mode. If any error is equal to or
                greater than TOL, is_correct is false
    DELTA (float): The difference parameter for the finite difference computation
    X1 (Tensor): size (48 x 2) denoting 72 example inputs each with 2 features
    X2 (Tensor): size (48 x 2) denoting the targets

    Returns
    -------
    is_correct (boolean): True if and only if MeanSquaredError passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx1 (float): the  error between the analytical and numerical gradients w.r.t X1
                    2. dzdx2 (float): The error between the analytical and numerical gradients w.r.t X2
    Note
    -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%
    dataset = torch.load("mean_squared_error_test.pt")
    X1 = dataset["X1"]
    X2 = dataset["X2"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    mean_squared_error = MeanSquaredError.apply
    # %%% DO NOT EDIT ABOVE %%%

    # analytical gradient
    y = mean_squared_error(X1, X2)
    z = torch.mean(y)
    z.backward()
    
    dzdy = torch.autograd.grad(z, y)

    # numerical gradient
    with torch.no_grad():
        # dzdx1
        dzdx1 = torch.zeros(X1.shape[0], X1.shape[1])
        for t in range(X1.shape[0]):
            for i in range(X1.shape[1]):
                # x+
                xp1 = X1.clone()
                xp1[t, i] += DELTA
                # x-
                xn1 = X1.clone()
                xn1[t, i] -= DELTA

                # second term
                v = (mean_squared_error(xp1, X2) - mean_squared_error(xn1, X2))/(2*DELTA)
                # dzdx1
                dzdx1[t, i] = torch.sum(torch.mul(dzdy[0], v))
        
        # dzdx2
        dzdx2 = torch.zeros(X2.shape[0], X2.shape[1])
        for t in range(X2.shape[0]):
            for i in range(X2.shape[1]):
                # x+
                xp2 = X2.clone()
                xp2[t, i] += DELTA
                # x-
                xn2 = X2.clone()
                xn2[t, i] -= DELTA

                # second term
                v = (mean_squared_error(X1, xp2) - mean_squared_error(X1, xn2))/(2*DELTA)
                # dzdx2
                dzdx2[t, i] = torch.sum(torch.mul(dzdy[0], v))

    # results
    is_correct = True
    err = {'dzdx1': 0, 'dzdx2': 0}
    err['dzdx1'] = torch.max(torch.abs(dzdx1 - X1.grad))
    err['dzdx2'] = torch.max(torch.abs(dzdx2 - X2.grad))

    # print(dzdx1)
    # print(X1.grad)
    for p in err:
        if err[p] >= TOL:
            is_correct = False

    # gradcheck
    gradcheck = torch.autograd.gradcheck(mean_squared_error, (X1, X2), eps=DELTA, atol=TOL)
    if not gradcheck:
        is_correct = False

    print(is_correct)
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = mean_squared_error_test()
    assert tests_passed
    print(errors)
