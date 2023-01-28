from fully_connected import FullyConnected
import torch


def fully_connected_test():
    """
    Provides Unit tests for the FullyConnected autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL (float): The error tolerance for the backward mode. If the error >= TOL, then is_correct is false
    DELTA (float): The difference parameter for the finite difference computations
    X (Tensor): of size (48 x 2), the inputs
    W (Tensor): of size (2 x 72), the weights
    B (Tensor): of size (72), the biases

    Returns
    -------
    is_correct (boolean): True if and only iff FullyConnected passes all unit tests
    err (Dictionary): with the following keys
                    1. dzdx: the error between the analytical and numerical gradients w.r.t X
                    2. dzdw (float): ... w.r.t W
                    3. dzdb (float): ... w.r.t B

    Note
    ----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%% DO NOT EDIT BELOW %%%%
    dataset = torch.load("fully_connected_test.pt")
    X = dataset["X"]
    W = dataset["W"]
    B = dataset["B"]
    TOL = dataset["TOL"]
    DELTA = dataset["DELTA"]
    full_connected = FullyConnected.apply
    # %%% DO NOT EDIT ABOVE

    # analytical gradient
    y = full_connected(X, W, B)
    z = torch.mean(y)
    z.backward()
    
    dzdy = torch.autograd.grad(z, y)

    # numerical gradient
    with torch.no_grad():
        # dzdx
        dzdx = torch.zeros(X.shape[0], X.shape[1])
        for t in range(X.shape[0]):
            for i in range(X.shape[1]):
                # x+
                xp = X.clone()
                xp[t, i] += DELTA
                # x-
                xn = X.clone()
                xn[t, i] -= DELTA

                # second term
                v = (full_connected(xp, W, B) - full_connected(xn, W, B))/(2*DELTA)
                # dzdx
                dzdx[t, i] = torch.sum(torch.mul(dzdy[0], v))
        
        # dzdw
        dzdw = torch.zeros(W.shape[0], W.shape[1])
        for t in range(W.shape[0]):
            for i in range(W.shape[1]):
                # w+
                wp = W.clone()
                wp[t, i] += DELTA
                # w-
                wn = W.clone()
                wn[t, i] -= DELTA

                # second term
                v = (full_connected(X, wp, B) - full_connected(X, wn, B))/(2*DELTA)
                # dzdw
                dzdw[t, i] = torch.sum(torch.mul(dzdy[0], v))

        # dzdb
        dzdb = torch.zeros(B.shape[0])
        for k in range(B.shape[0]):
                # w+
                bp = B.clone()
                bp[k] += DELTA
                # w-
                bn = B.clone()
                bn[k] -= DELTA

                # second term
                v = (full_connected(X, W, bp) - full_connected(X, W, bn))/(2*DELTA)
                # dzdb
                dzdb[k] = torch.sum(torch.mul(dzdy[0], v))

    # gradcheck
    gradcheck = torch.autograd.gradcheck(full_connected, (X, W, B) , eps=DELTA, atol=TOL)

    # results
    is_correct = True
    err = {'dzdx': 0, 'dzdw': 0, 'dzdb': 0}
    err['dzdx'] = torch.max(torch.abs(dzdx - X.grad))
    err['dzdw'] = torch.max(torch.abs(dzdw - W.grad))
    err['dzdb'] = torch.max(torch.abs(dzdb - B.grad))

    for p in err:
        if err[p] >= TOL:
            is_correct = False
    
    if not gradcheck:
        is_correct = False

    torch.save([is_correct , err], 'fully_connected_test_results.pt')
    return is_correct, err


if __name__ == '__main__':
    tests_passed, errors = fully_connected_test()
    assert tests_passed
    print(errors)

# %%
