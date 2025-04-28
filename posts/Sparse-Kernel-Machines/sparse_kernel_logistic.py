import torch

class KernelLogisticRegression:

    
    def __init__(self, kernel_fn, lam=0.1, gamma=1.0):
        """
        Initialize the Kernel Logistic Regression model.
        kernel_fn: callable: the kernel function to be used.
        lam: float: the regularization parameter.
        gamma: float: the kernel bandwidth parameter.
        """
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma
        self.a = None
        self.Xt = None


    def set_kernel_fn(self, kernel_fn):
        """
        Set the kernel function to be used.
        kernel_fn: callable: the kernel function to be used.
        """
        self.kernel_fn = kernel_fn
    
    def set_lam(self, lam):
        """
        Set the regularization parameter.
        lam: float: the regularization parameter.
        """
        self.lam = lam
    
    def set_gamma(self, gamma):
        """
        Set the kernel bandwidth parameter.
        gamma: float: the kernel bandwidth parameter.
        """
        self.gamma = gamma
    


    def score(self, X, recompute_kernel=False):
        """
        Compute the score for each data point in the feature matrix X. The score for the ith data point is a scalar value.
        If recompute_kernel is True, the kernel matrix is recomputed. Otherwise, the kernel matrix is reused.
        ARGUMENTS:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.
        recompute_kernel: bool: whether to recompute the kernel matrix or not.
        RETURNS:
            score, torch.Tensor: vector of scores. score.size() = (n,)
        """
        K = self.kernel_fn(X, self.Xt, self.gamma) if recompute_kernel else self.kernel_fn(X, self.Xt, self.gamma)
        return K @ self.a

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
            The prediction is 1 if the score is greater than 0, otherwise it is 0.
        """
        K = self.kernel_fn(X, self.Xt, self.gamma)
        score = K @ self.a
        return (score > 0).float()
    
    def loss(self, X, y):
        """
        Compute the loss, or the negative log-likelihood, for the Kernel Logistic Regression model.
        Arguments:
            X, torch.Tensor: the feature matrix. X.size() == (n, p),
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
    
    def fit(self, X, y, m_epochs=1000, lr=0.01):
        """
        Fit the Kernel Logistic Regression model to the training data.
        Arguments:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            m_epochs: int: the number of epochs to train the model.
            lr: float: the learning rate for the gradient descent optimization.
        """

        self.Xt = X
        n = X.shape[0]
        self.a = torch.zeros(n, requires_grad=False)

        for epoch in range(m_epochs):
            K = self.kernel_fn(X, self.Xt, self.gamma)
            s = K @ self.a
            sig = torch.sigmoid(s)
            grad = (sig - y) @ K + self.lam * torch.sign(self.a)
            self.a -= lr * grad


class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initialize the Gradient Descent Optimizer.
        model: KernelLogisticRegression: the model to be optimized.
        """
        self.model = model

    def step(self, X, y, lr=0.01):
        """
        Perform a single step of gradient descent.
        Arguments:
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}

            lr: float: the learning rate for the gradient descent step.
        """
        if self.model.a is None or self.model.Xt is None:
            self.model.Xt = X
            n = X.shape[0]
            self.model.a = torch.zeros(n, requires_grad=False)
        
        k = self.model.kernel_fn(X, self.model.Xt, self.model.gamma)
        s = k @ self.model.a
        sig = torch.sigmoid(s)
        grad = (sig - y) @ k + self.model.lam * torch.sign(self.model.a)
        self.model.a -= lr * grad




    

    
