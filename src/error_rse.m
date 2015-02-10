function rse = error_rse(w_hat, X, y)

%% calculate the error in predictions
% w_hat : D*1 vector of weights 
% X : NxD
% y : Nx1


%% calculations ahead
actuals = y;
predicted = w_hat' * X';
sqerr_vector= (actuals - predicted').^2;
rse = sqrt(sum(sqerr_vector));

%% End of File
