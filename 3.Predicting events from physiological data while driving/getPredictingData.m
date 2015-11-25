function data = getPredictingData()
% load raw ECG, GSR, labels data and preprocess them 
%% 1. Load raw ECG and GSR data and store them seperately
% addpath(genpath('TrainingAndTestingData/'));
% addpath(genpath('Neural Network/'));
numFiles = 27;% The number of total subjects
fileDir = 'PredictingData/';
fileName = 's';
fileType = 'csv';% The file type of raw data file
figureType = 'pdf';% The figure type
display =0; % We won't display the figure and save it.
GSRyRange = 12;
[Time_Minutes,GSR,ECG] = rawDataLoadDisplay(numFiles,fileDir, fileName, fileType, figureType,GSRyRange, display); 
clear fileDir;
clear fileName;
clear fileType;
clear display;
%% 2. Clean the GSR data in order to replace missing GSR data with the mean value of corresponding subjects. 
GSR_cleaned = cleanGSRDataForPredicting(GSR);
% createGSRECGfigure(2, GSR_cleaned{2},ECG{2},figureType);
% createGSRECGfigure(4, GSR_cleaned{4},ECG{4},figureType);
% createGSRECGfigure(5, GSR_cleaned{5},ECG{5},figureType);
% Drop subject 7
GSR_cleaned_dropped = {};
ECG_dropped = {};
for i = 1:numFiles-1
    if i <7
        GSR_cleaned_dropped{i} = GSR_cleaned{i};
        ECG_dropped{i} = ECG{i};
    else
        GSR_cleaned_dropped{i} = GSR_cleaned{i+1};
        ECG_dropped{i} = ECG{i+1};
    end
end
GSR_cleaned = GSR_cleaned_dropped;
ECG = ECG_dropped;
clear figureType;
%% 3. Filter GSR and ECG data in order to eliminate noise
M = 4;
b = ones(M,1)/M;
GSR_filtered_normalised = GSR_cleaned;
for i= 1:numFiles-1
    tempECG_filtered = filter(b,1,ECG{i});
    ECG_filtered_normalised{i} = normaliseFeatures(tempECG_filtered);
    tempGSR_filtered = filter(b,1,GSR_cleaned{i});
    GSR_filtered_normalised{i} = normaliseFeatures(tempGSR_filtered);
end 
clear tempECG_filtered;
clear tempGSR_filtered;
clear i; clear M; clear b;
%% Here we don't Smooth the GSR data with moving average method
% window_size = 10;% this a 2 second interval
% simple = tsmovavg(GSR_cleaned{1},'s',window_size,1);
% simple = simple(10:length(simple));
% clear window_size;
%% 3. Feature generation for each subject and target label process with 4 seconds intervel
secondsPerSeg = 4;%4 seconds interval
samplingRate = 5;% Sampling rate is 5 HZ
segOverlapProportion = 0.5;% 50% overlap for two next intervals
for i= 1:numFiles-1
    eachSubjects= retrieveFeatures(ECG_filtered_normalised{i}, GSR_filtered_normalised{i},...
                                    secondsPerSeg,samplingRate,segOverlapProportion);
    features_normalised{i} = normaliseFeatures(eachSubjects);
end 
overlapLength = secondsPerSeg*samplingRate*segOverlapProportion;
clear i;
clear eachSubjects;
clear overlapLength;
clear secondsPerSeg;
clear samplingRate;
clear segOverlapProportion;
%% 4. Conbine all subjects' features together and normalise them
featureMatrix = [];
for i= 1:numFiles-1
    featureMatrix = [featureMatrix',features_normalised{i}']';
end
clear i;
% clear features_normalised;
featureMatrix_normalised= normaliseFeatures(featureMatrix);
clear featureMatrix;
clear numFiles;
labels = zeros(length(featureMatrix_normalised),1);
data = [labels,featureMatrix_normalised];

