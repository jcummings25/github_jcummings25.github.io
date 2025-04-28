
import torch
class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1],))
        # your computation here: compute the vector of scores s
        return X @ self.w

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        return (self.score(X) > 0).float()
    
class LogisticRegression(LinearModel):
    
    def loss(self, X, y):
        """
        Compute the loss, or the negative log-likelihood, for the Logistic Regression model.
        
        Arguments: X, torch.Tensor: the feature matrix. X.size() == (n, p),
        where n is the number of data points and p is the
        number of features. This implementation always assumes
        that the final column of X is a constant column of 1s.

        y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

        Returns:
        loss, torch.Tensor: the average negative log-likelihood loss.
        
        """
        epsilon = 1e-7
        sigmoid = torch.sigmoid(self.score(X))
        return (-y * torch.log(sigmoid+epsilon) - (1 - y) * torch.log(1 - sigmoid + epsilon)).mean()
    
    def grad(self, X, y):
        """
        Compute the gradient of the loss function with regard to the weights w of the model.

        Arguments: X, torch.Tensor: the feature matrix. X.size() == (n, p),
        where n is the number of data points and p is the
        number of features. This implementation always assumes
        that the final column of X is a constant column of 1s.

        y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

        Returns: the mean of the gradients for each data point.
        """
        sig = torch.sigmoid(self.score(X))
        grad = (sig - y)[:, None] * X
        return grad.mean(0)

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
        self.prev_w = None

    def step(self, X, y, alpha, beta):
        """
        Perform a single step of gradient descent with momentum (if beta > 0)
        
        Arguments: X, torch.Tensor: the feature matrix. X.size() == (n, p),
        where n is the number of data points and p is the
        number of features. This implementation always assumes
        that the final column of X is a constant column of 1s.

        y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        alpha, float: the learning rate for the gradient descent step.

        beta, float: the momentum coefficient.
        """
        grad = self.model.grad(X, y)

        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        momentum = beta * (self.model.w - self.prev_w)
        new_w = self.model.w - alpha * grad + momentum
        self.prev_w = self.model.w.clone()
        self.model.w = new_w
        


