function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   http://networkrepository.com/pima-indians-diabetes.php
%   and splits the dataset into two sets: training set and testing set
%

 % ====================== YOUR CODE HERE ======================
    % Instructions: Import spreadsheets data, extract the first
    % 8 columns and store them as X. Extract the last column and 
    % store it as y. 
    %
    % Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    %
    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    %

    table = readtable("pima-indians-diabetes.csv");
    data = table2array(table);
    x = data(:, 1:8);
    y = data(:, end);

    tableHeight = height(x);
    randomRows = randperm(tableHeight);

    X_train = x(randomRows(1:floor(0.7*tableHeight)), :);
    X_test = x(randomRows(floor(0.7*tableHeight)+1:end), :);
    y_train = y(randomRows(1:floor(0.7*tableHeight)), :);
    y_test = y(randomRows(floor(0.7*tableHeight)+1:end), :);
    
    


% ============================================================
end