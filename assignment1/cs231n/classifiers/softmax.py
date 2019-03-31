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
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        exp_sums = np.log(np.sum(np.exp(scores)))
        loss += (-1 * y[i]) + exp_sums
        dW[:, y[i]] -= X[i, :].T 
        # dW[:, np.arange(num_classes)!=y[i]] += X[i, :].T
        for j in range(num_classes):
            if (j != y[i]):
                dW[:, j] += X[i, :].T
    # Compute average loss over all training samples by dividing by the total
    # number of training samples.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    N = X.shape[0]
    scores_exp = np.exp(scores)
    scores_norm = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)

    y_scores_norm = scores_norm[range(N), y]
    y_scores_log = -np.log(y_scores_norm)

    cross_entropy_loss = np.sum(y_scores_log) / N
    reg_loss = 0.5 * reg * np.sum(W*W)

    loss = cross_entropy_loss + reg_loss

    dscores = scores_norm
    dscores[range(N), y] -= 1
    dscores /= N

    dW = X.T.dot(dscores)
    dW += reg * W
    
    '''
    #correct_scores = -1 * scores[range(num_train), y]
    #sums_exp = np.log(np.sum(np.exp(scores), axis=1))
    softmax_X = np.exp(scores) / np.sum(np.exp(scores), axis=0)
    cross_entropy_loss = -np.log(softmax_X)
    #cross_entropy_loss = np.sum(correct_scores + sums_exp) / num_train
    reg_loss = reg * np.sum(W * W)
    loss = cross_entropy_loss + reg_loss
    '''
    #dscores = softmax_X
    #dscores[range(num_train), y] -= 1
    #dW = X.T.dot(dscores)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
