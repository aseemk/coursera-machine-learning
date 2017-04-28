function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Vectorized implementation from the lecture notes:
% https://www.coursera.org/learn/machine-learning/resources/Zi29t
% > "Simplified Cost Function and Gradient Descent" — unregularized!
h = sigmoid(X * theta);
J_unreg = (1 / m) * (-y' * log(h) - (1 - y)' * log(1 - h));
% > "Partial derivate of J(Θ)" -- unregularized!
grad_unreg = (1 / m) * X' * (h - y);

% Now regularize. Vectorized algorithm via tutorials:
% https://www.coursera.org/learn/machine-learning/discussions/weeks/3/threads/0DKoqvTgEeS16yIACyoj1Q
theta(1) = 0;
J = J_unreg + (lambda / (2 * m)) * theta' * theta;
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/GVdQ9vTdEeSUBCIAC9QURQ
grad = grad_unreg + (lambda / m) * theta;

% =============================================================

end
