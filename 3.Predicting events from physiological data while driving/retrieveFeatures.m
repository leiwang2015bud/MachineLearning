function features = retrieveFeatures(ecg, gsr,secondsPerSeg,samplingRate,segOverlapProportion)
% This function would generate features for each subject
% for each subject, there are several 5-second segment
% we would generate features for each segment
% Then we combine them together
%% 0.Extreme values for segments
% secondsPerSeg = 4;% 4 seconds interval
% samplingRate = 5;% Sampling rate is 5 HZ
% segOverlapProportion = 0.5;% 50% overlap for two next intervals
segLength = secondsPerSeg * samplingRate;% the number of Sampling points

%% 1.Generate features for GSR
filteredGSR = gsr(1:length(gsr)-1,1);
%% 1.1 Use filteredGSR data to generate features
[iqrValsGSR, medValsGSR,...
 meanValsGSR,rmsValsGSR,...
 stdValsGSR,minValsGSR,...
 maxValsGSR,rangeValsGSR] = getStatistics(filteredGSR, segLength, segOverlapProportion);

%% 1.2 Use gradient GSR data to generate features
[iqrValsGSRGrad, medValsGSRGrad,...
 meanValsGSRGrad,rmsValsGSRGrad,...
 stdValsGSRGrad,minValsGSRGrad,...
 maxValsGSRGrad,rangeValsGSRGrad] = getStatistics(gradient(filteredGSR), segLength, segOverlapProportion);

%% 2.Generate features for ECG data
filteredECG = ecg(1:length(gsr)-1,1);
%% 2.1 Use filteredECG data to generate features
[freqRatioValsECG,lfValsECG,hfValsECG] = getFrequencyRatio(filteredECG,secondsPerSeg, segLength, segOverlapProportion);
[iqrValsECG, medValsECG,...
 meanValsECG,rmsValsECG,...
 stdValsECG,minValsECG,...
 maxValsECG,rangeValsECG] = getStatistics(filteredECG, segLength, segOverlapProportion);
%% 3. Get features (27)
features = [iqrValsGSR, medValsGSR, meanValsGSR,rmsValsGSR, stdValsGSR,minValsGSR,maxValsGSR,rangeValsGSR,...%8    1.1 GSR features
iqrValsGSRGrad, medValsGSRGrad, meanValsGSRGrad,rmsValsGSRGrad, stdValsGSRGrad,minValsGSRGrad,maxValsGSRGrad,rangeValsGSRGrad,...%8/16   1.2 Gradient GSR features
freqRatioValsECG,lfValsECG,hfValsECG,...%3/19   2.1 ECG features (frequency related)
iqrValsECG,medValsECG, meanValsECG,rmsValsECG, stdValsECG,minValsECG,maxValsECG,rangeValsECG,...%8/27    2.1 ECG features 
];
% %% 2.2 Use maximum and minimum peaks range to generate features
% if(sum(ecg) > 0)
%     % 1) Calculate several maximum peaks 
%     peaksECG = getPeaksWithFindPeaks(ecg,segLength,segOverlapProportion);
%     peaksECG = peaksECG(zscore(peaksECG(:,2)) > -3,:);
%     % 2) Calculate several mimimum peaks
%     minPeaksECG = getPeaksWithFindPeaks(-ecg,segLength,segOverlapProportion);
%     minPeaksECG = minPeaksECG(zscore(minPeaksECG(:,2)) < 3,:);
%     % 3) Calculate intervel based on maximum and minimum peaks
%     tmpPeaks  = sortrows([peaksECG;[minPeaksECG(:,1),-minPeaksECG(:,2)]],1);
%     tmpPeakRanges = abs(diff(tmpPeaks(:,2)));   
%     % disp('Outliers - ECG');
%     % sum(zscore(tmpPeakRanges) > -2)
%     % length(tmpPeaks(2:length(tmpPeakRanges),1))
%     % length(tmpPeakRanges)    
%     tmpPeakRanges = [tmpPeaks(2:length(tmpPeaks),1),tmpPeakRanges];
%     tmpPeakRanges = tmpPeakRanges((zscore(tmpPeakRanges(:,2)) > -2) & (zscore(tmpPeakRanges(:,2)) < 5),:);
%     % figure;
%     % subplot(2,1,1);
%     % plot(tmpPeakRanges(:,1),tmpPeakRanges(:,2));
%     % subplot(2,1,2);
%     % plot(tmpPeakRanges(:,1),zscore(tmpPeakRanges(:,2)));   
%     % [tmpPeakRanges(:,2),zscore(tmpPeakRanges(:,2))]
%     % 4) Generate features with peaks range
%     tmpPeakRanges()
%     [iqrValsRangesECG, medValsRangesECG,...
%      meanValsRangesECG,rmsValsRangesECG,...
%      stdValsRangesECG,minValsRangesECG,...
%      maxValsRangesECG,rangeValsRangesECG] = getStatistics(tmpPeakRanges, segLength, segOverlapProportion);   
% end
% 
% %% 2.3 Use variability of ECG calculated from the maximum peaks to generate features
% if(sum(ecg) > 0)  
%     % 0) Get the variability of ECG calculated from the maximum peaks
%     [iqrVals, medVals,...
%      meanVals,rmsVals,...
%      stdVals,minVals,...
%      maxVals,rangeVals,...
%      peaksPerSegmentECG] = getStatistics(peaksECG, segLength, segOverlapProportion);
%     variabilityECG = [peaksECG(2:length(peaksECG),1),diff(peaksECG(:,1)),diff(peaksECG(:,2))];
%     % 1) Use variability of ECG to generate features including
%     % ECG variability features, ECG features, and HRV gradient stats features.
%     [freqRatioValsECGVar,lfValsECGVar,hfValsECGVar] = getFrequencyRatio([variabilityECG(:,1),abs(variabilityECG(:,3))],secondsPerSeg, segLength, segOverlapProportion);
%     [iqrValsVarECG, medValsVarECG,...
%      meanValsVarECG,rmsValsVarECG,...
%      stdValsVarECG,minValsVarECG,...
%      maxValsVarECG,rangeValsVarECG] = getStatistics([variabilityECG(:,1),variabilityECG(:,3)],segLength, segOverlapProportion);
%  
%     [iqrValsHRVGradStats, medValsHRVGradStats,...
%      meanValsHRVGradStats,rmsValsHRVGradStats,...
%      stdValsHRVGradStats,minValsHRVGradStats,...
%      maxValsHRVGradStats,rangeValsHRVGradStats] = getStatistics([variabilityECG(:,1),gradient(variabilityECG(:,3),variabilityECG(:,1))], segLength, segOverlapProportion);
%     % 2) Use variability of ECG to generate features including
%     % HRV features, time stats features, and HRV time gradient stats features.
%     [freqRatioValsHRV,lfValsHRV,hfValsHRV] = getFrequencyRatio([variabilityECG(:,1),variabilityECG(:,2)],secondsPerSeg, segLength, segOverlapProportion); 
%     [iqrValsVarECGTimeStats, medValsVarECGTimeStats,...
%      meanValsVarECGTimeStats,rmsValsVarECGTimeStats,...
%      stdValsVarECGTimeStats,minValsVarECGTimeStats,...
%      maxValsVarECGTimeStats,rangeValsVarECGTimeStats] = getStatistics([variabilityECG(:,1),variabilityECG(:,2)], segLength, segOverlapProportion);
%  
%     [iqrValsHRVTimeGradStats, medValsHRVTimeGradStats,...
%      meanValsHRVTimeGradStats,rmsValsHRVTimeGradStats,...
%      stdValsHRVTimeGradStats,minValsHRVTimeGradStats,...
%      maxValsHRVTimeGradStats,rangeValsHRVTimeGradStats] = getStatistics([variabilityECG(:,1),gradient(variabilityECG(:,2),variabilityECG(:,1))], segLength, segOverlapProportion);    
%     % Activities of SNS and PNS are depicted within the intervals of 0.05-0.15
%     % Hz (LF) and 0.15-0.4 Hz (HF)
%     % find LF/HF
%     % freqECGVals = getFrequencyRatio(filteredECG,secondsPerSeg, segLength, segOverlapProportion);
% end
%% Feature matrix: there are 74 features in total
% features = [
% iqrValsGSR, medValsGSR, meanValsGSR,rmsValsGSR, stdValsGSR,minValsGSR,maxValsGSR,rangeValsGSR,...%8    1.1 GSR features
% iqrValsGSRGrad, medValsGSRGrad, meanValsGSRGrad,rmsValsGSRGrad, stdValsGSRGrad,minValsGSRGrad,maxValsGSRGrad,rangeValsGSRGrad,...%8/16   1.2 Gradient GSR features
% freqRatioValsECG,lfValsECG,hfValsECG,...%3/19   2.1 ECG features (frequency related)
% iqrValsECG,medValsECG, meanValsECG,rmsValsECG, stdValsECG,minValsECG,maxValsECG,rangeValsECG,...%8/27    2.1 ECG features 
% iqrValsRangesECG, medValsRangesECG, meanValsRangesECG,rmsValsRangesECG, stdValsRangesECG,minValsRangesECG,maxValsRangesECG,rangeValsRangesECG,...%8/35  2.2 ECG features generated from maximum and minimum peaks range
% peaksPerSegmentECG,...%1/36      2.3 0)feature is generated from the variability of ECG calculated from the maximum peaks
% freqRatioValsECGVar,lfValsECGVar,hfValsECGVar,... %3/39    2.3 1) ECG features
% iqrValsVarECG, medValsVarECG, meanValsVarECG,rmsValsVarECG, stdValsVarECG,minValsVarECG,maxValsVarECG,rangeValsVarECG,...%8/47    ECG features,
% iqrValsHRVGradStats, medValsHRVGradStats, meanValsHRVGradStats,rmsValsHRVGradStats, stdValsHRVGradStats,minValsHRVGradStats,maxValsHRVGradStats,rangeValsHRVGradStats,...%8/55    HRV gradient stats features.
% freqRatioValsHRV,lfValsHRV,hfValsHRV,... %3/58      2.3 2)  HRV features
% iqrValsVarECGTimeStats, medValsVarECGTimeStats, meanValsVarECGTimeStats,rmsValsVarECGTimeStats, stdValsVarECGTimeStats,minValsVarECGTimeStats,maxValsVarECGTimeStats,rangeValsVarECGTimeStats,...%8/66    time stats features
% iqrValsHRVTimeGradStats, medValsHRVTimeGradStats, meanValsHRVTimeGradStats,rmsValsHRVTimeGradStats, stdValsHRVTimeGradStats,minValsHRVTimeGradStats,maxValsHRVTimeGradStats,rangeValsHRVTimeGradStats,... %8/74   HRV time gradient stats features
% ];
end