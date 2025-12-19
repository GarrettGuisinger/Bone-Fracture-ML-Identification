function accuracy = evaluateAccuracy(beta1, beta2, X, y)
%EVALUATEACCURACY calculates the prediction accuracy of the learned 
%neural network model using the testing data 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the percentage of accurately predicted examples 
%
%
p = predict(beta1, beta2, X);
accuracy = (sum(p == y) / height(y)) * 100;



% ============================================================

end