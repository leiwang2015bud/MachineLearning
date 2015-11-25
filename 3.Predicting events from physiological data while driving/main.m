%% Set training set and testing set into one data matrix 
data = getTrainingAndTestingData();
%% Training and Testing with K-fold validation
% Change the label category defination
classes = data(:,1);
classes = classes * -1.0;
classes(classes== -1) = 0;
classes(classes== -2) = 0;
classes(classes== -3) = 1;
classes(classes== -4) = 1;
classes(classes== -5) = 1;
classes(classes==-6) = 1;
% Chnage the class number into the required range for ELM
classes = classes+1;
dataFeatures = data(:,2:size(data,2));
data_ELM = [classes,dataFeatures];
% Select the classes which is larger than 'classesThreshold'
classesThreshold = 0;
data_ELM = data_ELM(data_ELM(:,1) > classesThreshold,:);
% initializes the CP object with whole instances target labels
cp = classperf(data_ELM(:,1)-classesThreshold); 
% Set parameters for Extreme Learning machine
Elm_Type = 1;% 1 means classification; 0 means regression
NumberofHiddenNeurons = 350;% the number of hidden Neurons
ActivationFunction = 'sig';% the activation function for training/testing/predicting neuron network model.
% Set parameters for k-fold validation
N = length(data_ELM);% The number of whole instances without separatation for train and test
indices = crossvalind('Kfold',N,10);
bestTestingAccuracy = 0;
for i = 1:10
    % There are 15 examples as testing set
    test = (indices == i); train = ~test;
    % so there would 15 classifier result for testing set
    [TrainingTime, TestingTime, ...
     TrainingAccuracy, TestingAccuracy, ...
     training_label_index_actual, testing_label_index_actual, ...
     InputWeight, OutputWeight, BiasofHiddenNeurons] = ELM(data_ELM(train,:), data_ELM(test,:),...
                                                        Elm_Type, NumberofHiddenNeurons,...
                                                        ActivationFunction);
    % trace the best model which have the highest testing accuracy
    if bestTestingAccuracy < TestingAccuracy
        bestTestingAccuracy = TestingAccuracy;
        bestInputWeight =InputWeight ;
        bestOutputWeight = OutputWeight;
        bestBiasofhiddenNeurons = BiasofHiddenNeurons;       
    end
    % updates the CP object with the current classification results
    classperf(cp,testing_label_index_actual,test);
end
score = cp.CorrectRate % queries for the correct classification rate
CountingMatrix	= cp.CountingMatrix	;
%% Predicting with the best model which have the best testing accuracy 
bestInputWeight =InputWeight ;
bestOutputWeight = OutputWeight;
bestBiasofhiddenNeurons = BiasofHiddenNeurons;

PredictingData = getPredictingData();

[PredictingTime,...
 predicting_label_index_actual] = Predicting_ELM( PredictingData,ActivationFunction,...
                                                  InputWeight, OutputWeight,...
                                                  BiasofHiddenNeurons);

predicting_labels = reshape(predicting_label_index_actual,[299,26]);