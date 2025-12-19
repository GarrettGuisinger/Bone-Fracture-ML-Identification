%% SDSU Machine Learning Course (EE510/EE600/CompE596)
%% Programming Assignment:  Neural Network 
%
%  Instructions
%  ------------
% 
%  Dataset comes from: 
%   http://networkrepository.com/pima-indians-diabetes.php
% 
% 
%  This file contains code that helps you get started on the
%  neural network assignment. 
%
%  You will need to complete the following functions in this 
%  assignment
%
%     loadData.m
%     featureNormalize.m
%     trainNN.m
%     evaluateAccuracy.m
%     predict.m
%     sigmoid.m
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).


% Initialization
clear ; close all; clc

%% =========== Part 1: Data Preprocessing =============
fprintf('===== Part 1: Data Preprocessing ===== \n');

% Instructions: The following code loads data into matlab, splits the 
%               data into two sets, and performs feature normalization. 
%               You will need to complete code in loadData.m, and 
%               featureNormalize.m
% ============================================================



fprintf('Loading data ...\n');

% ====================== YOUR CODE HERE ======================
% Step 1: Load data
[X_train, y_train, X_test, y_test] = loadData();

% ============================================================
% Print out some data points
fprintf('First 10 examples from the training dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n',...
    [X_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');

[num_train, m] = size(X_train); % num_train is the number of total data examples
                                % m is the number of features
                        
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

% Step 2: Normalize the features. 
% ====================== YOUR CODE HERE ======================
[Xn_train, mu, sigma] = featureNormalize(X_train);

% ============================================================
% Print out some normalized data points
fprintf('First 10 examples from the training dataset after normalization: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n',...
    [Xn_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 2: Train Neural Network ============

fprintf('===== Part 2: Train Neural Network ===== \n');

% Instructions: The following code applies backpropagation algorithm to 
%               estimate the parameters in a neural network model with 
%               a single hidden layer. 
%               You should complete code in trainNN.m, and sigmoid.m
%
%               Try running the backpropagation algorithm with 
%               different values of alpha and num_hidden and see which 
%               one gives you the best result.
% ============================================================


% ====================== YOUR CODE HERE ======================
% Step 1: Configure the hyper-parameters, including
%         1. alpha: learning rate
%         2. num_epochs: number of training epochs
%         3. num_hidden: number of hidden units (not including the bia unit)
%         4. num_output: number output unit

alpha = 0.01; % you should change this
num_epochs = 500; % you should change this 
num_hidden = 100;   % you should change this 
num_output = size(y_train,2);   % you should change this 

% initialize the weights: beta1 and beta2 
beta1 = rand(num_hidden, m+1); % weights associated with links between input and hidden layers
beta2 = rand(num_output, num_hidden+1); % weights associated with links between hidden and output layers
% ============================================================

% ====================== YOUR CODE HERE ======================
% Step 2: Run backpropagation 
[beta1, beta2, J_history] = trainNN(Xn_train, y_train, beta1, beta2, alpha, num_epochs);

% ============================================================

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of epochs');
ylabel('Cost J');

% Display gradient descent's result
fprintf('beta1 computed from gradient descent: \n');
fprintf(' %f \n', beta1);
fprintf('\n');
fprintf('beta2 computed from gradient descent: \n');
fprintf(' %f \n', beta2);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 3: Evaluate performance =============

fprintf('===== Part 3: Evaluate Performance ===== \n');

% Instructions: The following code evaluates the performance of
%               the trained neural network model. You should 
%               complete code in evaluateAccuracy.m, predict.m
% ============================================================


% ====================== YOUR CODE HERE ======================
% Hint: The testing set also needs to be normlized first
Xn_test = (X_test - mu) ./ sigma;
Xn_test = [ones(height(X_test), 1) Xn_test]; 
accuracy = evaluateAccuracy(beta1, beta2, Xn_test, y_test);

% ============================================================

% Display the prediction accuracy
fprintf('Accuracy:\n %f\n', accuracy);
fprintf('\n');

% ====================== YOUR CODE HERE ======================
% Step 2: Given a new input x = [3, 100, 79, 19, 100, 36, 0.8, 30], predict the output
x_new = [3, 100, 79, 19, 100, 36, 0.8, 30]
x_new = (x_new - mu) ./ sigma;
x_new = [1, x_new];

y_new = predict(beta1, beta2, x_new);

% ============================================================
% display the predicted output
fprintf(['Predicted output:\n %d\n'], y_new);


