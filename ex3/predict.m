function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Layer 1 is the input X. However, unlike the lecture where this layer was
% purely a column vector (of the n+1 features), our input is a matrix of
% (m rows) x (n columns). We want to maintain the m rows. Add zeros.
A1 = [ones(m, 1) X];

% Compute layer 2. This is normally g(z2) where z2 = Theta1 * a1, which
% results in a column vector of s2 rows (the number of units in layer 2).
% Since we want to maintain m rows, we want to get (m rows) * (s2 columns),
% so we transpose Theta1, which is originally (s2 rows) x (n+1 columns).
Z2 = A1 * Theta1';
A2 = sigmoid(Z2);

% Add zeros to layer 2 to compute layer 3.
A2 = [ones(m, 1) A2];

% Compute layer 3, which is our output layer, similarly.
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% Layer 3 is now an (m rows) x (k columns) matrix, where each column for a
% given row represents the probability that input example is that class.
% Like `predictOneVsAll`, we take the max prob class.
[dummy p] = max(A3, [], 2);

% =========================================================================


end
