
%Reset Screen
clear ; close all; clc


ytrain = dlmread('NUDataytrain.csv');
Xtrain = dlmread('NUDataXtrain.csv');

Xval = dlmread('NUDataXval.csv')
yval = dlmread('NUDatayval.csv');

%train model
theta = trainLinearReg(Xtrain,ytrain,0);

save 'Thetavalues.csv' theta;

m = size(Xtrain, 1);


%Test on the cross validation set
h = Xval * theta;
h = h > 0.5;
accuracy = sum(h == yval) / length(yval);

fprintf('Given no regularization and the basic 10 features, accuracy is estimated at %% %f \n', accuracy * 100);

[lambda_vec, error_train, error_val] = ...
    validationCurve(Xtrain, ytrain, Xval, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
  normTheta = trainLinearReg(Xtrain, ytrain, lambda_vec(i));
  h = Xval * normTheta;
  h = h > 0.5;
  accuracy = sum(h == yval) / length(yval);
  fprintf('Given regularization value of lambda = %f and the basic 10 features, accuracy is estimated at %% %f \n', lambda_vec(i), accuracy * 100);
end

##fprintf('\n \n Time for the learning curve... \n \n');
##
##lambda = 0;
##[error_train, error_val] = ...
##    learningCurve([ones(m, 1) Xtrain], ytrain, ...
##                  [ones(size(Xval, 1), 1) Xval], yval, ...
##                  lambda);
##
##plot(1:m, error_train, 1:m, error_val);
##title('Learning curve for linear regression')
##legend('Train', 'Cross Validation')
##xlabel('Number of training examples')
##ylabel('Error')
##axis([0 13 0 150])
##
##fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
##for i = 1:m
##    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
##end
