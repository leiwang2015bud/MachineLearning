function GSR_cleaned = cleanGSRDataForPredicting(GSR)
%%We would clean the GSR data in order to replace missing GSR data with 
% the mean value of corresponding subjects. 
% Since we have observe whole GSR data and find only subject 4,5,9 need to
% be cleaned.
%% Clean subject 4 GSR data
% use GSR{i} get the i-th column vector from big GSR cell
% Based on human observation, we would drop the GSR data between 2182 ~2209
GSR3_cleaned = cleanForOneSubject(1594, 1756, GSR{3});
GSR{3} = GSR3_cleaned;

GSR6_cleaned = cleanForOneSubjectEnds(2897, 3001, GSR{6},7.5);
GSR{6} = GSR6_cleaned;

GSR19_cleaned = cleanForOneSubject(2340, 2491, GSR{19});
GSR{19} = GSR19_cleaned;

GSR21_cleaned = cleanForOneSubjectEnds(2700, 3001, GSR{21},4);
GSR{21} = GSR21_cleaned;

GSR23_cleaned = cleanForOneSubject(878, 1172, GSR{23});
GSR{23} = GSR23_cleaned;

GSR26_cleaned = cleanForOneSubjectEnds(2930, 3001, GSR{26},2);
GSR{26} = GSR26_cleaned;

GSR_cleaned = GSR;

%% We define a subfunction to serve for clean GSR data for each subject
function GSRi_cleaned = cleanForOneSubject(startIndex, endIndex, GSRi)
GSRi_drop = GSRi(startIndex:endIndex);
% GSRi_rest = [GSRi(1:startIndex-1)',GSRi(endIndex+1:length(GSRi))']';
GSRi_drop_neighbour = [GSRi(startIndex-35:startIndex-1)',GSRi(endIndex+1:endIndex+35)']';
meani = mean(GSRi_drop_neighbour);
GSRi_drop = GSRi_drop.*0.0 + meani;
GSRi_cleaned = [GSRi(1:startIndex-1)',GSRi_drop',GSRi(endIndex+1:length(GSRi))']';
end

%% We define a subfunction to serve for clean GSR data for each subject
function GSRi_cleaned = cleanForOneSubjectEnds(startIndex, endIndex, GSRi,meani)
GSRi_drop = GSRi(startIndex:endIndex);
% GSRi_rest = [GSRi(1:startIndex-1)',GSRi(endIndex+1:length(GSRi))']';
GSRi_drop = GSRi_drop.*0.0 + meani;
GSRi_cleaned = [GSRi(1:startIndex-1)',GSRi_drop',GSRi(endIndex+1:length(GSRi))']';
end

end