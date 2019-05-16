function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_temp = [0.01 0.03 0.1 0.3 1 3 10 30 ];
sigma_temp = [0.01 0.03 0.1 0.3 1 3 10 30 ];
C_test = [];
sigma_test = [];
err=[];
for i = C_temp
    for j = sigma_temp
        
        model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
        predictions = svmPredict(model, Xval);
        err_1 = mean(double(predictions ~= yval));
        err = [err err_1];
        C_test = [C_test i];
        sigma_test = [sigma_test j];
    end 
end
error = [err; C_test; sigma_test];
[v,i]= min(err,[],2);
 ik = error(:,i);
 C = ik(2,1);
 sigma = ik(3,1);

% =========================================================================

end
