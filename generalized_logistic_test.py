from generalized_logistic import GeneralizedLogistic
import torch


def generalized_logistic_test():
    """
    Provides Unit tests for the GeneralizedLogistic autograd Function

    PROVIDED CONSTANTS
    ------------------
    TOL1 (float): the  error tolerance for the forward mode. If the error >= TOL1, is_correct is false
    TOL2 (float): The error tolerance for the backward mode
    DELTA (float): The difference parameter for the finite differences computation
    X (Tensor): size (48 x 2) of inputs
    L, U, and G (floats): The parameter values necessary to compute the hyperbolic tangent (tanH) using
                        GeneralizedLogistic
    Returns:
    -------
    is_correct (boolean): True if and only if GeneralizedLogistic passes all unit tests
    err (Dictionary): with the following keys
                        1. y (float): The error between the forward direction and the results of pytorch's tanH
                        2. dzdx (float): the error between the analytical and numerical gradients w.r.t X
                        3. dzdl (float): ... w.r.t L
                        4. dzdu (float): ... w.r.t U
                        5. dzdg (float): .. w.r.t G
     Note
     -----
    The error between arbitrary tensors x and y is defined here as the maximum value of the absolute difference between
    x and y.
    """
    # %%%% DO NOT EDIT BELOW %%%
    dataset = torch.load("generalized_logistic_test.pt")
    X = dataset["X"]
    L = dataset["L"]
    U = dataset["U"]
    G = dataset["G"]
    TOL1 = dataset["TOL1"]
    TOL2 = dataset["TOL2"]
    DELTA = dataset["DELTA"]
    generalized_logistic = GeneralizedLogistic.apply
    # %%%  DO NOT EDIT ABOVE %%%

    # analytical gradient
    y = generalized_logistic(X, L, U, G)
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
                v = (generalized_logistic(xp, L, U, G) - generalized_logistic(xn, L, U, G))/(2*DELTA)
                # dzdx
                dzdx[t, i] = torch.sum(torch.mul(dzdy[0], v))
        
        # dzdl
        lp = L.clone()      # l+
        lp += DELTA
        ln = L.clone()      # l-
        ln -= DELTA
        v = (generalized_logistic(X, lp, U, G) - generalized_logistic(X, ln, U, G))/(2*DELTA)   # second term
        dzdl = torch.sum(torch.mul(dzdy[0], v))

        # dzdu
        up = U.clone()      # u+
        up += DELTA
        un = U.clone()      # u-
        un -= DELTA
        v = (generalized_logistic(X, L, up, G) - generalized_logistic(X, L, un, G))/(2*DELTA)   # second term
        dzdu = torch.sum(torch.mul(dzdy[0], v))

        # dzdg
        gp = G.clone()      # u+
        gp += DELTA
        gn = G.clone()      # u-
        gn -= DELTA
        v = (generalized_logistic(X, L, U, gp) - generalized_logistic(X, L, U, gn))/(2*DELTA)   # second term
        dzdg = torch.sum(torch.mul(dzdy[0], v))

    # results
    is_correct = True
    err = {'dzdx': 0, 'dzdl': 0, 'dzdu': 0, 'dzdg': 0, 'y': 0}

    # forward check with TOL1
    err['y'] = torch.max(torch.abs(torch.tanh(X) - y))
    if err['y'] >= TOL1:
        is_correct = False

    # gradcheck with TOL2
    gradcheck = torch.autograd.gradcheck(generalized_logistic, (X, L, U, G) , eps=DELTA, atol=TOL2)
    if not gradcheck:
        is_correct = False

    # backward check with TOL2
    err['dzdx'] = torch.max(torch.abs(dzdx - X.grad))
    err['dzdl'] = torch.max(torch.abs(dzdl - L.grad))
    err['dzdu'] = torch.max(torch.abs(dzdu - U.grad))
    err['dzdg'] = torch.max(torch.abs(dzdg - G.grad))
    for p in err:
        if err[p] >= TOL2:
            is_correct = False

    torch.save([is_correct , err], 'generalized_logistic_test_results.pt')
    return is_correct, err


if __name__ == '__main__':
    test_passed, errors = generalized_logistic_test()
    assert test_passed
    print(errors)
