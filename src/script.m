%%=========================================================================
% Course : Machine Learning CSE 574
% Project: PROJECT 2 -- learn linear regression
% Authors: 
%           ANGAD GADRE
%           HARISH MANGALAMPALLI
%           RAJARAM RABINDRANATH
%%=========================================================================


%% House keeping
clearvars;
path = ''; %% FIXME --- Add path
cd(path);
load diabetes;


%% ==== Adding bias(intercepts) column to the dataset
x_train_i = [ones(size(x_train,1),1) x_train];
x_test_i = [ones(size(x_test,1),1) x_test];


%%% FILL CODE FOR PROBLEM 1 %%%
% linear regression without intercept

%% =================== BEGIN PROBLEM 2 CODE =======================
%  Experiment OLS 
%  A: Without Intercept
%  B: With Intercept
%  ========================================================================

% A: OLE without the intercept
start_prob1_A = cputime;
w_hat = learnOLERegression(x_train,y_train);
timeTaken_prob1_A = cputime - start_prob1_A;

error_train = error_rse(w_hat,x_train,y_train);
error_test = error_rse(w_hat,x_test,y_test);

disp('OLE errors [sans intercept] -- train and test ');
disp(error_train)
disp(error_test)

% linear regression with intercept
start_prob1_B = cputime;
w_hat_i = learnOLERegression(x_train_i,y_train);
timeTaken_prob1_B = cputime - start_prob1_B;

error_train_i = error_rse(w_hat_i,x_train_i,y_train);
error_test_i = error_rse(w_hat_i,x_test_i,y_test);

disp('OLE errors [with intercept] -- train and test ');
disp(error_train_i)
disp(error_test_i)


weights_intercept_avg = sum(abs(w_hat_i))/size(w_hat_i,1);

%%% END PROBLEM 1 CODE %%%

%% =================== BEGIN PROBLEM 2 CODE =======================
%                   Experiment with ridge regression
%  ========================================================================

% ridge regression using loss function to estimate W_mle
lambdas = 0:0.00001:0.001;
prob2_train_errors = zeros(length(lambdas),1);
prob2_test_errors = zeros(length(lambdas),1);

start_prob2 =  cputime;
for i = 1:length(lambdas)
    l = lambdas(i);
    w_ridge = learnRidgeRegression(x_train_i,y_train,l);
    prob2_train_errors(i) = error_rse(w_ridge,x_train_i,y_train);
    prob2_test_errors(i) = error_rse(w_ridge,x_test_i,y_test);
end

timeTaken_prob2 = cputime - start_prob2;



figure;
hold on;
plot(lambdas,prob2_train_errors,'green');
plot(lambdas,prob2_test_errors,'blue');
legend('Training Error','Test Error');
title('Ridge Regression [Analytical Approach]- Train errors vs. Test errors')
xlabel('Lambdas')
ylabel('Errors')
hold off;

% optimal lambda -- the one with minimum error w.r.t in test data
[minVal,minValIndex] = min(prob2_test_errors);
lambda_optimal = lambdas(minValIndex);
minVal
w_ridge = learnRidgeRegression(x_train_i,y_train,lambda_optimal);
Loptimal_train_errors = error_rse(w_ridge,x_train_i,y_train);
Loptimal_test_errors = error_rse(w_ridge,x_test_i,y_test);

weights_ridge_avg = sum(abs(w_ridge))/size(w_ridge,1);

Loptimal_test_errors
Loptimal_train_errors

weights_intercept_avg
weights_ridge_avg
%% END PROBLEM 2 CODE


%% =================== BEGIN PROBLEM 3 CODE =======================
%            Ridge Regression learning using Gradient descent 
%  ========================================================================

% set the maximum number of iteration in conjugate gradient descent
lambdas = 0:0.00001:0.001;
initialWeights = zeros(65,1);
options = optimset('MaxIter', 500);

% define the objective function
prob3_train_errors= zeros(length(lambdas),1);
prob3_test_errors = zeros(length(lambdas),1);

start_gradDesc = cputime;  
for i = 1:length(lambdas)
    l = lambdas(i);
    objFunction = @(params) regressionObjVal(params, x_train_i, y_train, l);
    w = fmincg(objFunction, initialWeights, options); 
    %function rse = error_rse(w_hat, X, y)
    prob3_train_errors(i) = error_rse(w,x_train_i,y_train);
    prob3_test_errors(i) = error_rse(w,x_test_i,y_test);
end
timeTaken_gradDesc = cputime-start_gradDesc;

figure;
hold on;
plot(lambdas,prob3_train_errors,'green');
plot(lambdas,prob3_test_errors,'blue');
legend('Training Error','Test Error');
title('Ridge Regression [Gradient Descent]- Train errors vs. Test errors')
xlabel('Lambdas')
ylabel('Errors')
hold off;

% optimal lambda -- the one with minimum error w.r.t in test data
[minVal,minValIndex] = min(prob3_test_errors);
lambda_optimal_grad_desc = lambdas(minValIndex);
lambda_optimal_grad_desc
minVal

timeTaken_gradDesc
timeTaken_prob2
%% END PROBLEM 3 CODE

%% =================== BEGIN PROBLEM 4 CODE =======================
%       Non-linear regression -- for varying number of derived attr
%          All attr are derived from attr(3) of the orig dataset
%  ========================================================================

% using variable number 3 only
x_train_3 = x_train(:,3);
x_test_3 = x_test(:,3);

% for plots
dValues = 0:1:6;


prob4_train_errors_l_zero = zeros(7,1);
prob4_test_errors_l_zero = zeros(7,1);

% no regularization
l = 0;
start_prob4_A = cputime;
for d = 0:6
    x_train_n = mapNonLinear(x_train_3,d);
    x_test_n = mapNonLinear(x_test_3,d);
    w = learnRidgeRegression(x_train_n,y_train,l); % find the W_mle for given lambda
    prob4_train_errors_l_zero(d+1) = error_rse(w,x_train_n,y_train);
    prob4_test_errors_l_zero(d+1) = error_rse(w,x_test_n,y_test);
end
timeTaken_prob4_A = cputime - start_prob4_A;

figure;
hold on;
plot(dValues,prob4_train_errors_l_zero,'green');
plot(dValues,prob4_test_errors_l_zero,'blue');
legend('Training Error','Test Error');
title('Non-linear Regression - Train errors vs. Test errors [Lamda = 0]')
xlabel('D values')
ylabel('Errors')
hold off;

% get best value of d for lambda =  0
[minVal_lzero,minValIndex] = min(prob4_test_errors_l_zero);
best_d_zeroLambda = minValIndex-1;

prob4_train_errors_l_opti = zeros(7,1);
prob4_test_errors_l_opti = zeros(7,1);

% optimal regularization
l = lambda_optimal; % from part 2
start_prob4_B =  cputime;
for d = 0:6
    x_train_n = mapNonLinear(x_train_3,d);
    x_test_n = mapNonLinear(x_test_3,d);
    w = learnRidgeRegression(x_train_n,y_train,l); % find the W_mle for lambda_optimal
    prob4_train_errors_l_opti(d+1) = error_rse(w,x_train_n,y_train);
    prob4_test_errors_l_opti(d+1) = error_rse(w,x_test_n,y_test);
end
timeTaken_prob4_B = cputime-start_prob4_B;


figure;
hold on;
plot(dValues,prob4_train_errors_l_opti,'green');
plot(dValues,prob4_test_errors_l_opti,'blue');
legend('Training Error','Test Error');
title('Non-linear Regression - Train errors vs. Test errors [Lamda = OptLambda]')
xlabel('D values')
ylabel('Errors')
hold off;

% get best value of d for lambda = optimal[from ridge exp]
[minVal_lOpti,minValIndex] = min(prob4_test_errors_l_opti);
best_d_optLambda = minValIndex-1;

%=========================== fit a curve to the data
if(minVal_lOpti > minVal_lzero)
    fit_curvePoly = best_d_zeroLambda; 
else
   fit_curvePoly = best_d_optLambda;
end
    
dataset_train = mapNonLinear(x_train_3,fit_curvePoly);
dataset_test =  mapNonLinear(x_test_3,fit_curvePoly);
l =0;
w_nonLinear_lzero = learnRidgeRegression(dataset_train,y_train,l); 
w_nonLinear_lopti = learnRidgeRegression(dataset_train,y_train,lambda_optimal);

err_train_lzero= error_rse(w_nonLinear_lzero,dataset_train,y_train);
err_train_lopti = error_rse(w_nonLinear_lopti,dataset_train,y_train);

y_lzero = w_nonLinear_lzero'*dataset_train';
y_lopt = w_nonLinear_lopti'*dataset_train';

figure;
hold on;
plot(x_train_3,y_train,'o');
plot(x_train_3,y_lopt,'r');
plot(x_train_3,y_lzero,'g');
legend('Diabetes levels','LambdaOpt','LambdaZero','Location','southeast');
title('Non-linear Regression - Curve/Line fitting for D=1[Train Data]')
xlabel('X')
ylabel('Diabetes Level')
hold off;

y_lzero = w_nonLinear_lzero'*dataset_test';
y_lopt = w_nonLinear_lopti'*dataset_test';



err_test_lzero= error_rse(w_nonLinear_lzero,dataset_test,y_test);
err_test_lopti = error_rse(w_nonLinear_lopti,dataset_test,y_test);

figure;
hold on;
plot(x_test_3,y_test,'o');
plot(x_test_3,y_lopt,'r');
plot(x_test_3,y_lzero,'g');
legend('Diabetes levels','LambdaOpt','LambdaZero','Location','southeast');
title('Non-linear Regression - Curve/Line fitting for D=1[Test Data]')
xlabel('X')
ylabel('Diabetes Level')
hold off;

%errors
err_train_lzero
err_train_lopti

err_test_lzero
err_test_lopti

% weights:
w_nonLinear_lzero
w_nonLinear_lopti


%% END OF PROBLEM 4 code

%% time taken
timeTaken_prob1_A
timeTaken_prob1_B
timeTaken_prob2
timeTaken_gradDesc
timeTaken_prob4_A
timeTaken_prob4_B

%% PACKING ALL RESULTS
save('results_prj2.mat', 'error_test', 'error_test_i', 'error_train',...
'error_train_i', 'lambda_optimal','lambda_optimal_grad_desc','w_hat','w_hat_i',...
'w_ridge','best_d_zeroLambda','best_d_optLambda','timeTaken_prob1_A',...
'timeTaken_prob1_B','timeTaken_prob2','timeTaken_gradDesc');

%% END OF FILE
