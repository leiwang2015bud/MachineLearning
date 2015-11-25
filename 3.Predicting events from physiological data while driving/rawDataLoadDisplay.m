function [Time_Minutes,GSR,ECG] = rawDataLoadDisplay(numFiles,fileDir, fileName, fileType, figureType,GSRyRange,  display)
%% Input values:
% numFiles = 1;
% fileDir = '/Users/bud/Desktop/ Driving Simulator Experiment/GSR&ECG/TrainingAndTestingData'
% fileName = 's'
% fileType = 'csv';
% figureType = 'pdf';    %'pdf' Full page Portable Document Format (PDF) color	.pdf
% figureType = 'tiffn' %'tiffn'	TIFF 24-bit (not compressed).tif
% figureType = 'eps'   %'eps'	Encapsulated PostScript?(EPS) Level 3 black and white	.eps
% figureType = 'epsc'  %'epsc'	Encapsulated PostScript (EPS) Level 3 color	.eps
%% Output values:
% Time_Minutes is a 3001 x 1 cell
% GSR is a 1 x numFiles cell
% ECG is a 1 x numFiles cell
%% A for loop to read each file and plot the GSR ECG figure with storing each figure into specific figure type
for i= 1:numFiles
    [Time_Minutes,GSR{i},ECG{i}] = readGSRECG([fileDir,fileName, num2str(i),'.',fileType], 2, 3002);
    if display ~= 0 % if display Not equal to 0
        createGSRECGfigure(i, GSR{i},ECG{i},figureType, GSRyRange);
    end
end   