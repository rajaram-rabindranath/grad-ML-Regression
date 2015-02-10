function [error, error_grad] = regressionObjVal(w, X, y, lambda)

% compute squared error (scalar) and gradient of squared error with respect
% to w (vector) for the given data X and y and the regularization parameter
% lambda

%% we know the loss function - lets comput the loss and the loss_grad

samples = size(X,1);
features = size(X,2);

%% error is computed using the loss function
actuals = y;                    % a Nx1 vector
X_trans = X';                   % DxN matrix [X = NxD matrix]
predicted = (w' * X_trans)';    % [1xD * DxN]' = Nx1
reg_params = lambda*(sum(w.^2));% Scalar 1x1
error = (sum((actuals - predicted).^2))/samples + reg_params;

%% error_grad :: is calc by finding the grad of the loss function 
error_grad = 2*(((predicted - actuals)'*X)./(samples)); % derivative of firts component of loss fn
error_grad = error_grad'+ lambda*2*w; % derivative of the second component of the loss fn

end
%% End of File
