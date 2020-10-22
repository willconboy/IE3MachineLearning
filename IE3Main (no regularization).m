%CODE USED TO TRAIN ALGORITHM

%Reset Screen
clear ; close all; clc


%Import data
data = load('nudata.csv');

[m, n] = size(data);

n = n - 1;

X = data(:, [1:n])
y = data(:, n + 1)

%Initialize theta to zeros
initial_theta = zeros(n + 1,1);

size(initial_theta)

%Normalize the X values
##X = [featureNormalize(X(:,1:n-3)) X(:, n-3 : n-1)];

%Add in bias unit
X = [ones(m, 1) X];

%Split X into Xtrain(training) and Xval(cross validation)
%I am doing a 70/30 split because this is easier

cutoff = floor(m * 0.7); %% the cutoff value for the 70/30 split

%Shuffle X values (I do this to avoid Xtrain being 2010-2017)
shuffleorder = randperm(size(X, 1));
Xtemp = X(shuffleorder, :);
ytemp = y(shuffleorder, :);

Xtrain = Xtemp([1:cutoff], :);
ytrain = ytemp([1:cutoff], :);
Xval = Xtemp([cutoff:end], :);
yval = ytemp([cutoff:end], :);


%% Something we could (or should) do is save cross validation and training sets 
%% as indiviudal files. The current way I'm doing it trains the model *slightly*
%% differently every time. It shouldn't really matter, but consistency is nice


%train model
theta = trainLinearReg(Xtrain,ytrain,0)
size(theta);

%Test on the cross validation set
h = Xval * theta;
h = h > 0.5;
accuracy = sum(h == yval) / length(yval);

fprintf('Given no regularization and the basic 10 features, accuracy is estimated at %% %f \n', accuracy * 100);


##%%Now lets save the data for future consistency
##save 'NUDataXval.txt' Xval;
##save 'NUDatayval.txt' yval;
##save 'NUDataXtrain.txt' Xtrain;
##save 'NUDataytrain.txt' ytrain;
