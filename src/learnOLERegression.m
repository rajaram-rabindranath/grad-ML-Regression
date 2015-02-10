function w = learnOLERegression(X,y)

%% Implement OLE training here
% Inputs:
% X = N x D
% y = N x 1
% Output:
% w = D x 1

%% will implement the following 
% inverse{(X_trans X)} * (X_trans * y)

%X = x_train;
%y = y_train;

X_trans = X';
X_sqr = X_trans * X;
inter_right = X_trans * y;
w = X_sqr\inter_right;

%w_try = inv(X_sqr)* inter_right;

%z = w - w_try;
%sum(z)
%% END OF FILE