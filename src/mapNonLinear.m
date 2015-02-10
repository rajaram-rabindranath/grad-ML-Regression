function x_n = mapNonLinear(x,d)
% Inputs:
% x - a single column vector (N x 1)
% d - integer (>= 0)
% Outputs:
% x_n - (N x (d+1))

%% create the dataset that shall be used for fitting


% add new derived columns to the original data vectors
x_n = []; % make the vector null
for n=0:d
newCol = x.^n;
x_n = horzcat(x_n,newCol);
end


end
%% END OF FILE