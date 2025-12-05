from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.       #
    ###########################################################################
    # Reshape input into rows: (N, D)
    N = x.shape[0]
    x_row = x.reshape(N, -1)  # flatten each example

    # Compute output: out = xW + b
    out = x_row.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass. Store results in dx, dw, db.  #
    ###########################################################################
    # Same reshape as in affine_forward
    N = x.shape[0]
    x_row = x.reshape(N, -1)

    # d(out)/d(w):   x_row^T @ dout  → (D, M)
    dw = x_row.T.dot(dout)

    # d(out)/d(b): sum over batch dimension N → (M,)
    db = dout.sum(axis=0)

    # d(out)/d(x_row): dout @ w^T → (N, D)
    dx_row = dout.dot(w.T)

    # Reshape back to original x shape: (N, d1, ..., dk)
    dx = dx_row.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass. Store the result in out.         #
    ###########################################################################
    # ReLU: output = max(0, x)
    out = np.maximum(0, x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass. Store the result in dx.         #
    ###########################################################################
    # Gradient is passed only where x > 0
    dx = dout * (x > 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Implement softmax loss and gradient. Store them in loss and dx.   #
    ###########################################################################
    # Number of samples
    N = x.shape[0]

    # ----- Forward: compute softmax loss -----

    # Numeric stability: shift scores by max per row
    shifted_logits = x - np.max(x, axis=1, keepdims=True)  # (N, C)

    # exp of scores
    exp_scores = np.exp(shifted_logits)  # (N, C)

    # probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

    # cross-entropy loss: -log p(correct class)
    correct_logprobs = -np.log(probs[np.arange(N), y])  # (N,)
    loss = np.sum(correct_logprobs) / N

    # ----- Backward: gradient w.r.t. scores x -----

    dx = probs.copy()  # (N, C)
    dx[np.arange(N), y] -= 1  # subtract 1 for correct class
    dx /= N  # average over batch

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # Mini-batch statistics
        mu = np.mean(x, axis=0)  # (D,)
        x_centered = x - mu  # (N, D)
        var = np.var(x, axis=0)  # (D,)
        std = np.sqrt(var + eps)  # (D,)

        # Normalize
        x_hat = x_centered / std  # (N, D)

        # Scale and shift
        out = gamma * x_hat + beta  # (N, D)

        # Update running estimates
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var

        # Cache values needed for backward
        cache = (x, x_hat, mu, var, std, gamma, beta, eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Normalize using running statistics
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
        cache = (x, x_hat, running_mean, running_var, np.sqrt(running_var + eps), gamma, beta, eps)
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # Unpack cache (from training forward pass)
    x, x_hat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    # Gradients wrt beta and gamma
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient wrt normalized activations
    dx_hat = dout * gamma

    # Backprop through normalization: x_hat = (x - mu) / std
    x_centered = x - mu

    # d(x_centered) from division by std
    dx_centered_div = dx_hat / std

    # dstd from dependence in denominator
    dstd = -np.sum(dx_hat * x_centered, axis=0) / (std ** 2)

    # Convert dstd to dvar: std = sqrt(var + eps)
    dvar = dstd * 0.5 / std

    # d(x_centered) from var path: var = mean(x_centered^2)
    dx_centered_var = (2.0 / N) * x_centered * dvar

    # Total gradient wrt centered inputs
    dx_centered = dx_centered_div + dx_centered_var

    # Gradient wrt mean: x_centered = x - mu
    dmu = -np.sum(dx_centered, axis=0)

    # Finally gradient wrt inputs
    dx = dx_centered + dmu / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Unpack cache (training path)
    x, x_hat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    # Parameter gradients
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient wrt normalized activations
    dx_hat = dout * gamma

    # Closed-form input gradient
    sum_dx_hat = np.sum(dx_hat, axis=0)  # (D,)
    sum_dx_hat_xhat = np.sum(dx_hat * x_hat, axis=0)  # (D,)
    dx = (dx_hat * N - sum_dx_hat - x_hat * sum_dx_hat_xhat) / (N * std)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # LayerNorm: normalize per data point (across features)
    # Compute per-sample mean/var along axis=1
    mu = np.mean(x, axis=1, keepdims=True)  # (N, 1)
    var = np.var(x, axis=1, keepdims=True)  # (N, 1)
    std = np.sqrt(var + eps)  # (N, 1)

    x_hat = (x - mu) / std  # (N, D)
    out = gamma * x_hat + beta  # (N, D)

    cache = (x, x_hat, mu, var, std, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # Unpack cache
    x, x_hat, mu, var, std, gamma, beta, eps = cache
    N, D = x.shape

    # Parameter gradients (sum over batch dimension)
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient wrt normalized activations
    dx_hat = dout * gamma  # (N, D)

    # Closed-form input gradient for LayerNorm (normalize over features)
    sum_dx_hat = np.sum(dx_hat, axis=1, keepdims=True)  # (N, 1)
    sum_dx_hat_xhat = np.sum(dx_hat * x_hat, axis=1, keepdims=True)  # (N, 1)
    dx = (dx_hat * D - sum_dx_hat - x_hat * sum_dx_hat_xhat) / (D * std)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # Inverted dropout: keep each unit with probability p
        # Scale kept activations by 1/p so expectation stays the same
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # 1. Extract parameters
    stride = conv_param['stride']
    pad = conv_param['pad']

    # 2. Get shapes
    N, C, H, W = x.shape  # batch, channels, height, width
    F, _, HH, WW = w.shape  # filters, channels, filter height/width

    # 3. Pad the input on height and width dimensions
    #    np.pad(array, pad_width, mode)
    # pad_width is a tuple for each dimension: (before, after)
    x_padded = np.pad(
        x,
        ((0, 0),  # no padding for N
         (0, 0),  # no padding for C
         (pad, pad),  # pad 'pad' rows at top & bottom
         (pad, pad)),  # pad 'pad' columns at left & right
        mode='constant'
    )

    # 4. Compute output spatial size
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # 5. Allocate output
    out = np.zeros((N, F, H_out, W_out))

    # 6. Perform the convolution
    for n in range(N):  # over images
        for f in range(F):  # over filters
            for i in range(H_out):  # vertical position in output
                for j in range(W_out):  # horizontal position in output
                    # top-left corner of the window in the padded input
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # Extract the patch: shape (C, HH, WW)
                    x_patch = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # Element-wise multiply and sum, then add bias
                    out[n, f, i, j] = np.sum(x_patch * w[f]) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # Unpack cache
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Shapes
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape

    # Pad input as in forward pass
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode='constant'
    )

    # Allocate gradients
    dx_padded = np.zeros_like(x_padded)  # grad w.r.t padded x
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Backprop: loop over all positions
    for n in range(N):  # each image
        for f in range(F):  # each filter
            for i in range(H_out):  # output height index
                for j in range(W_out):  # output width index
                    # same window coordinates as forward
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    # the input patch that produced out[n,f,i,j]
                    x_patch = x_padded[n, :, h_start:h_end, w_start:w_end]

                    # upstream gradient (scalar)
                    dout_curr = dout[n, f, i, j]

                    # gradient w.r.t. bias: sum over all dout
                    db[f] += dout_curr

                    # gradient w.r.t. filter weights
                    # note: x_patch and w[f] have same shape (C, HH, WW)
                    dw[f] += x_patch * dout_curr

                    # gradient w.r.t. padded input
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout_curr

    # Remove padding from dx_padded to get dx (original x shape)
    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # 1. Read parameters from dict
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # 2. Input shape
    N, C, H, W = x.shape

    # 3. Compute output spatial dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # 4. Allocate output
    out = np.zeros((N, C, H_out, W_out))

    # 5. Max-pooling
    for n in range(N):  # over images
        for c in range(C):  # over channels
            for i in range(H_out):  # output height index
                for j in range(W_out):  # output width index
                    # window coordinates in the input
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # extract patch
                    x_patch = x[n, c, h_start:h_end, w_start:w_end]

                    # take maximum in this window
                    out[n, c, i, j] = np.max(x_patch)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # 1. Unpack cache
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # 2. Shapes
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    # 3. Initialize gradient w.r.t. input
    dx = np.zeros_like(x)

    # 4. Backward pass: loop over windows
    for n in range(N):  # each image
        for c in range(C):  # each channel
            for i in range(H_out):  # output height index
                for j in range(W_out):  # output width index
                    # window coordinates in input
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # the input patch that was pooled
                    x_patch = x[n, c, h_start:h_end, w_start:w_end]

                    # where was the max?
                    max_val = np.max(x_patch)
                    mask = (x_patch == max_val)  # boolean array same shape as patch

                    # upstream gradient for this pooled output (scalar)
                    dout_curr = dout[n, c, i, j]

                    # distribute gradient only to max locations
                    dx[n, c, h_start:h_end, w_start:w_end] += mask * dout_curr
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = x.shape

    # Step 1: reshape x into 2D: (N*H*W, C)
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)

    # Step 2: apply vanilla batch normalization
    out_bn, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)

    # Step 3: reshape output back to (N, C, H, W)
    out = out_bn.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape

    # Step 1: reshape dout to match forward reshaping
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)

    # Step 2: call vanilla batchnorm backward
    dx_bn, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    # Step 3: reshape dx back to spatial format
    dx = dx_bn.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N, C, H, W = x.shape

    # Step 1: reshape into groups
    x_group = x.reshape(N, G, C // G, H, W)

    # Step 2: compute per-group mean and variance
    mean = x_group.mean(axis=(2, 3, 4), keepdims=True)
    var = x_group.var(axis=(2, 3, 4), keepdims=True)

    # Step 3: normalize
    x_hat = (x_group - mean) / np.sqrt(var + eps)

    # Step 4: reshape back to (N, C, H, W)
    x_hat = x_hat.reshape(N, C, H, W)

    # Step 5: scale and shift (broadcast applies correctly)
    out = gamma * x_hat + beta

    # Store values for backward
    cache = (G, x, x_hat, mean, var, eps, gamma, beta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    G, x, x_hat, mean, var, eps, gamma, beta = cache
    N, C, H, W = x.shape
    group_size = C // G

    # 1) dgamma and dbeta
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

    # 2) dx_hat
    dx_hat = dout * gamma

    # 3) reshape into groups
    dx_hat_group = dx_hat.reshape(N, G, group_size, H, W)
    x_group = x.reshape(N, G, group_size, H, W)
    m = group_size * H * W

    # 4) backward through normalization
    dvar = np.sum(dx_hat_group * (x_group - mean),
                  axis=(2, 3, 4), keepdims=True) * \
           (-0.5) * (var + eps) ** (-3 / 2)

    dmean = np.sum(dx_hat_group * (-1 / np.sqrt(var + eps)),
                   axis=(2, 3, 4), keepdims=True) + \
            dvar * np.sum(-2 * (x_group - mean),
                          axis=(2, 3, 4), keepdims=True) / m

    dx_group = dx_hat_group / np.sqrt(var + eps) + \
               dvar * 2 * (x_group - mean) / m + \
               dmean / m

    # 5) reshape back
    dx = dx_group.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
