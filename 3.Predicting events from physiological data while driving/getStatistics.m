function [iqrVals, medVals,...
          meanVals,rmsVals,....
          stdVals,minVals,...
          maxVals,rangeVals,...
          peaksPerSegment] = getStatistics(inData, segLength, segOverlapProportion)
% Inputs:
% inData in {ecg, gsr} for each subject
% nParaTypes = segLength = length ?
% segOverlapProportion = 0.5
% features are not found for times when there is a transition from one type
% of paragraph to the next
overlapLength = segLength * segOverlapProportion;
numOfInstanceRows = 1+(length(inData)-segLength)/overlapLength;

% 1) minimum values
%   For vectors, MIN(X) is the smallest element in X. For matrices,
%   MIN(X) is a row vector containing the minimum element from each
%   column. For N-D arrays, MIN(X) operates along the first
%   non-singleton dimension.
%   Example: If X = [2 8 4   then min(X,[],1) is [2 3 4],
%                    7 3 9]
%
%       min(X,[],2) is [2    and min(X,5) is [2 5 4
%                       3],                   5 3 5].
minVals = zeros(numOfInstanceRows,1);
% 2) maximum value
maxVals = zeros(numOfInstanceRows,1);
% 3)For vectors, MEDIAN(a) is the median value of the elements in a.
%   For matrices, MEDIAN(A) is a row vector containing the median
%   value of each column. http://en.wikipedia.org/wiki/Median
medVals = zeros(numOfInstanceRows,1);
% 4)Get the interquartile range of the values
%   the length of rectangular in box plot
iqrVals = zeros(numOfInstanceRows,1);
% 5)Get mean value
meanVals = zeros(numOfInstanceRows,1);%pingjunzhi
% 6)Get the standard deviation.  For matrices,
% 7)stdVals is a row vector containing the standard deviation of each column.
stdVals = zeros(numOfInstanceRows,1);% 
varVals = zeros(numOfInstanceRows,1);% stdVals.^2 We didn't output this feature
% 8)Get the Root Mean Square
rmsVals = zeros(numOfInstanceRows,1);
% 9) Get the peaks for each intervel
peaksPerSegment = zeros(numOfInstanceRows,1);

row = 1;
for t1=1:overlapLength:(length(inData) - segLength + 1)
    t2 = t1 + segLength - 1;
    dataSeg = inData(t1:t2);        
    if(~isempty(dataSeg))
        minVals(row) = min(dataSeg);
        iqrVals(row) = iqr(dataSeg);
        medVals(row) = median(dataSeg);
        stdVals(row) = std(dataSeg);
        varVals(row) = var(dataSeg);
        maxVals(row) = max(dataSeg);
        meanVals(row) = mean(dataSeg);
        rmsVals(row) = sqrt(mean(dataSeg.^2));
        peaksPerSegment(row) = length(dataSeg);
    else
        minVals(row) = NaN;
        iqrVals(row) = NaN;
        medVals(row) = NaN;
        stdVals(row) = NaN;
        varVals(row) = NaN;
        maxVals(row) = NaN;
        meanVals(row) = NaN;
        rmsVals(row) = NaN;
        peaksPerSegment(row) = NaN;
     end
     row = row + 1;
     
     
end
% minLimit = (numOfInstanceRows*0.75);
% 
% if(sum(minVals==0) > minLimit)
%     minVals = [];
% end
% if(sum(iqrVals==0) > minLimit)
%     iqrVals = [];
% end
% if(sum(medVals==0) > minLimit)
%     medVals = [];
% end
% if(sum(stdVals==0) > minLimit)
%     stdVals = [];
% end
% if(sum(varVals==0) > minLimit)
%     varVals = [];
% end
% if(sum(maxVals==0) > minLimit)
%     maxVals = [];
% end
% if(sum(meanVals==0) > minLimit)
%     meanVals = [];
% end
% if(sum(rmsVals==0) > minLimit)
%     rmsVals = [];
% end
% if(sum(peaksPerSegment==0) > minLimit)
%     peaksPerSegment = [];
% end
rangeVals = maxVals - minVals;
