function [beta1, beta2, J_history] = trainNN(X, y, beta1, beta2, alpha, num_epochs)
%TRAINNN train the neural network model using backpropagation algorithm. It
%updates the weights, beta1 and beta2 using the training examples. It also
%generates the cost computed after each epoch. 

% useful values
[n, ~] = size(X); % n is number of training examples
num_hidden = length(beta1(:,1)); % number of hidden units (bias not included)
num_output = length(beta2(:,2)); % number of output units

J_history = zeros(num_epochs,1); % stores value of the cost function J at each iteration

for epoch = 1:num_epochs
% for each training example, do the following
    Jd = 0;
    for d = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the weights beta1 and
    %               beta2. The key steps are indicated as follows
    %
    %
    
 
        %% Step 1: forward propagate to generate the network output

        training_input = X(d, :);
        training_output = y(d);

        a1 = [1 training_input];
        o1 = sigmoid(a1 * beta1');
        a2 = [1 o1];
        o2 = sigmoid(a2 * beta2');
        
       
        

        
        %% Step 2: for each output unit, calculate its error term
        % Recall that the number of output units is num_output

        output_error = o2 .* (1 - o2) .* (training_output - o2);
        
        
        
        %% Step 3: for each hidden unit, calculate its error term
        % Recall that number of hidden units is num_hidden+1
 
        hidden_error = o1 .* (1 - o1) .* (output_error * beta2(:, 2:end));
        
        

        %% Step 4: update the weights using the error terms
        
        beta1 = beta1 + alpha * (hidden_error' * a1);
        beta2 = beta2 + alpha * (output_error' * a2);
        
        
        
        %% calculate the cost (Jd = SSE) per epoch
        Jd = Jd + sum((o2 - training_output)^2);
    end
    J_history(epoch) = Jd/(2*n);
end