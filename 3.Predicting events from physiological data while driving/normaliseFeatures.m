function features_normalised= normaliseFeatures(features)
% Normalise the features Matrix to the range of [-1,1]
features_normalised = features;
for i=1:size(features,2)
    % if the range of each raw feature is in proper range
    % we scale them into [-1,1]
    if max(features(:,i))~=min(features(:,i))
        features_normalised(:,i)=(features(:,i)-min(features(:,i)))/(max(features(:,i))-min(features(:,i)))*2-1;
    %otherwise, feature in little range change or without range change 
    %we replace this feature with a column of 0.
    else
        features_normalised(:,i)=0;
    end
end
end
