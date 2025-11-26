from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means;
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize

        # LOSS
        logp = np.log(p)
        loss -= logp[y[i]]  # negative log probability is the loss

        # Grad ADDED LINES
        p[y[i]] -= 1.0
        dW += np.outer(X[i], p)
        #      ADDED LINES


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W                         # ADDED LINE


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    N = X.shape[0]
    dW = np.zeros_like(W)
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute softmax probabilities
    exp_scores = np.exp(scores)
    p = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Loss: average cross-entropy + regularization
    loss = -np.sum(np.log(p[np.arange(N), y])) / N
    loss += reg * np.sum(W * W)

    p[np.arange(N), y] -= 1
    p /= N

    # Gradient w.r.t. W + regularization gradient
    dW = X.T.dot(p)
    dW += 2 * reg * W


    return loss, dW
