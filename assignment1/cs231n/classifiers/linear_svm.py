import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # get scores for ith example
        scores = X[i].dot(W)
        # store score for correct class of ith example in a variable
        correct_class_score = scores[y[i]]
        # iterate over scores for each class for ith example
        num_pos = 0
        for j in range(num_classes):
            # if j is indexing the correct class for the ith example, continue to next iteration
            if (j != y[i]):
                # compute margin for jth class score
                margin = scores[j] - correct_class_score + 1 # note delta = 1
                # adds margin to total loss of margin > 0
                
                if (margin > 0):
                    loss += margin
                    # scale dW at correct class column negatively by input values of ith example to descend gradient
                    # [3072 : 1]  -= [1: 3072].T 
                    # scale dW at current class column positively by input values of ith example 
                    #dW[:, y[i]] -= X[i, :].T
                    # [3072 : 1] += [1 : 3072].T
                    dW[:, j] += X[i, :].T
                    num_pos += 1
            else:
                dW[:, y[i]] += -num_pos * X[i, :].T 

    # Average out loss and gradient since computed over all training instances
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss and gradient
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W # derivative of regularization formula
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # Half vectorized:
    scores = X.dot(W)
    num_train = X.shape[0]
    correct_scores = scores[range(num_train), y]
    margins = np.maximum(0, scores - correct_scores[:,None] + 1.0)
    margins[range(num_train), y] = 0

    data_loss = np.sum(margins) / num_train
    reg_loss = reg * np.sum(W * W)
    loss = data_loss + reg_loss


    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    num_pos = np.sum(margins > 0, axis=1) # number of positive losses
    dscores = np.zeros(scores.shape)
    dscores[margins > 0] = 1
    dscores[range(num_train), y] = -num_pos

    dW = X.T.dot(dscores) / num_train
    dW += 2 * reg * W
    #############################################################################
    #                              END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
