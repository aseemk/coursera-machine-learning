function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Convert our y vector of m {1, ... k} labels into an m x k matrix
% of 0-1 values! Via help from the tutorial:
% https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/threads/QFnrpQckEeWv5yIAC00Eog
y_matrix = eye(num_labels)(y,:);

% Adapted from the vectorized logistic regression cost function (ex 3)
% and vectorized feed-forward algorithm from `predict.m` & tutorial:
a1 = X; % m x input_layer_size
z2 = [ones(m, 1) a1] * Theta1'; % m x hidden_layer_size
a2 = sigmoid(z2);   % our hidden unit activations, also m x hidden_layer_size
z3 = [ones(m, 1) a2] * Theta2'; % m x num_labels
a3 = sigmoid(z3);   % our output unit activations, also m x num_labels
% TODO: Is there a way to vectorize this loop? I don't yet understand this:
% https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/AzIrrO7wEeaV3gonaJwAFA
% Fortunately, we expect `num_labels` to be pretty small, so not a big deal?
for k = 1:num_labels
    % Extract the current label's vectors from the matrices:
    y_k = y_matrix(:,k);
    h_k = a3(:,k);
    % Then increment by the vectorized cost function part for this label:
    J = J + (1 / m) * (-y_k' * log(h_k) - (1 - y_k)' * log(1 - h_k));
end

% Now add regularization. Adapted from ex 3 and tutorial again, except
% NOTE: We cannot compute the sums of squares via matrix multiplication!
% So using double-sums over element-wise multiplication instead.
% I *think* this is what the above thread explains.
Theta1_unbiased = Theta1(:,2:end);
Theta2_unbiased = Theta2(:,2:end);
J = J + (lambda / (2 * m)) * (sum(sum(Theta1_unbiased .^ 2)) + sum(sum(Theta2_unbiased .^ 2)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

% Following the vectorized tutorial:
% https://www.coursera.org/learn/machine-learning/discussions/all/threads/a8Kce_WxEeS16yIACyoj1Q
d3 = a3 - y_matrix; % m x num_labels
d2 = (d3 * Theta2_unbiased) .* sigmoidGradient(z2); % m x hidden_layer_size
Delta1 = d2' * [ones(m, 1) a1]; % hidden_layer_size x (input_layer_size + 1)
Delta2 = d3' * [ones(m, 1) a2]; % num_labels x (hidden_layer_size + 1)
Theta1_grad = (1 / m) .* Delta1;
Theta2_grad = (1 / m) .* Delta2;

% Now add regularization, again via the tutorial:
Theta1(:,1) = 0;    % hidden_layer_size x (input_layer_size + 1)
Theta2(:,1) = 0;    % num_labels x (hidden_layer_size + 1)
Theta1_grad  = Theta1_grad + (lambda / m) .* Theta1;
Theta2_grad  = Theta2_grad + (lambda / m) .* Theta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% NOTE: Already done above.


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
