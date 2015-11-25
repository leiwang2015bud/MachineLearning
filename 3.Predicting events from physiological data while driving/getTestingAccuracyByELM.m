function score = getTestingAccuracyByELM(featureIndeices,data)
% input:
% 1. featureIndeices: a logical vector whose size is (nfeatures,1)
% 2. data: the whole data whose first column is label, and the rest are features
% Output:
% score: the testing accuracy
%% Training and Testing with K-fold validation
% Set feature matrix with feature indeices and target labels
[nRols, nCol] = size(data);
nFeats = nCol -1;
% featureIndeices = true(nFeats,1);% a logical indeices where 1 represent we choos this feature
featureIndeices = logical(featureIndeices);
dataFeatures = data(:,2:size(data,2));
dataFeatures = dataFeatures(:,featureIndeices);
dataLabel = data(:,1);
data = [dataLabel,dataFeatures];
% Set parameters for Extreme Learning machine
Elm_Type = 1;% 1 means classification; 0 means regression
NumberofHiddenNeurons = 300;% the number of hidden Neurons
ActivationFunction = 'sig';% the activation function for training/testing/predicting neuron network model.
% Set parameters for k-fold validation
N = length(data);% The number of whole instances without separatation for train and test
indices = crossvalind('Kfold',N,10);
classes = data(:,1);
cp = classperf(classes); % initializes the CP object with whole instances target labels
for i = 1:10
    % There are 15 examples as testing set
    test = (indices == i); train = ~test;
    % so there would 15 classifier result for testing set
    [TrainingTime, TestingTime, ...
     TrainingAccuracy, TestingAccuracy, ...
     training_label_index_actual, testing_label_index_actual, ...
     InputWeight, OutputWeight, BiasofHiddenNeurons] = ELM(data(train,:), data(test,:),...
                                                        Elm_Type, NumberofHiddenNeurons,...
                                                        ActivationFunction);
    % updates the CP object with the current classification results
    classperf(cp,testing_label_index_actual,test);
end
score = cp.CorrectRate; % queries for the correct classification rate
end