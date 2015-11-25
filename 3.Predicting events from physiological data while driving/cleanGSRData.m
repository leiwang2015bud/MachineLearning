function GSR_cleaned = cleanGSRData(GSR)
%%We would clean the GSR data in order to replace missing GSR data with 
% the mean value of corresponding subjects. 
% Since we have observe whole GSR data and find only subject 4,5,9 need to
% be cleaned.
%% Clean subject 4 GSR data
% use GSR{i} get the i-th column vector from big GSR cell
% Based on human observation, we would drop the GSR data between 2182 ~2209
GSR4_cleaned = cleanForOneSubject(2180, 2226, GSR{4});
GSR{4} = GSR4_cleaned;

GSR5_cleaned = cleanForOneSubject(2551, 2675, GSR{5});
GSR{5} = GSR5_cleaned;

GSR2_cleaned = cleanForOneSubject(528, 716, GSR{2});
GSR{2} = GSR2_cleaned;

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

end