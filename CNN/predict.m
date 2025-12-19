function p = predict(beta1, beta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(beta1, beta2, X) outputs the probability of the output to 
%   1, given input X and trained weights of a neural network (beta1, beta2)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. 
%
%
[n, ~] = size(X);
for d = 1:n

    data_input = X(d, :);

    o1 = sigmoid(data_input * beta1');
    a2 = [1 o1];
    o2 = sigmoid(a2 * beta2');

    p(d) = (o2 >= 0.5);
end

% =========================================================================


end
