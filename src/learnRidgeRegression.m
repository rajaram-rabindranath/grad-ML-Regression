function w = learnRidgeRegression(X,y,lambda)

% Implement ridge regression training here
% Inputs:
% X = N x D
% y = N x 1
% lambda = scalar
% Output:
% w = D x 1
%% We know the loss function -- need to minimize the loss function
% We shall find the w = W_mle given the parameters that we have

samples = size(X,1);                        
I = eye(size(X,2));                         % DxD identity matrix
X_trans = X';                               % DxN matrix [X is NxD matrix]
X_sqr = X_trans * X;                        % a DxD matrix [DxN * NxD]
right_inter = X_trans * y;                  % Dx1 vector [DxN * Nx1]
left_inter = (samples*lambda*I + X_sqr);    % a DxD matrix 
w = left_inter\right_inter;                 % Dx1 vector of weights

end
%% END OF FILE
