function [freqRatioVals,lfVals,hfVals] = getFrequencyRatio(inData,secondsPerSeg, segLength, segOverlapProportion)
HF_low = 0.15;
LF_low = 0.05;
HF_high = 0.4;
LF_high = 0.15;

overlapLength = segLength * segOverlapProportion;
numOfFeatureRows = 1+(length(inData)-segLength)/overlapLength;

freqRatioVals = zeros(numOfFeatureRows,1);
lfVals = zeros(numOfFeatureRows,1);
hfVals = zeros(numOfFeatureRows,1);

row = 1;
for t1=1:overlapLength:(length(inData) - segLength + 1)
    t2 = t1 + segLength - 1;
    dataSeg = inData(t1:t2); 
    if(~isempty(dataSeg))
        N = length(dataSeg); %% number of points
        T = secondsPerSeg; %% define time of interval, We selected it as 4 seconds
        f = dataSeg; %%define function, 10 Hz sine wave
        p = abs(fft(f))/(N/2); %% absolute value of the fft
        p = p(1:0.5:N/2).^2; %% take the power of positve freq. half
        freq = (0:0.5:N/2-1)/T; %% find the corresponding frequency in Hz
        % semilogy(freq,p); %% plot on semilog scale
        % axis([0 20 0 1]); %% zoom in
        powers = [freq',p];
        if(isempty(powers))
            HFPower = NaN;
            LFPower = NaN;
        else
            HFPower = sum(powers(powers(:,1)> HF_low & powers(:,1)<= HF_high,2));
            LFPower = sum(powers(powers(:,1)>= LF_low & powers(:,1)< LF_high,2));
        end
        
        freqRatioVals(row) = LFPower/HFPower;
        lfVals(row) = LFPower;
        hfVals(row) = HFPower;
    end
    row = row + 1;

end
