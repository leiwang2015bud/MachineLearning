function [alpha,b,varargout]=dualfisher(K,y,gamma,varargin)

%function [alpha,ytest,varargout]=dualfisher(K,y,gamma,varargin)
%
% Dual (kernel) Fisher discriminant analysis (KFDA).
%
%INPUTS
% K = the kernel matrix on the training data (dimension ell x ell)
% gamma = the regularization parameter
% y = the training labels
% varargin = optional argument specifying Ktest (ell x elltest), the
%            matrix specifying the kernel between test and training
%            samples, and optionally also true test labels
% 
%OUTPUTS
% alpha = the dual vector specifying the FDA direction
% ytest = the label predictions on the test data
% varargout = optional output containing the label predictions ytest according
%             to Ktest, and potentially also the test error, only available
%             when the true test labels are specified in varargin
%
%
%For more info, see www.kernel-methods.net

% K is the kernel matrix of ell training points
% lambda the regularisation parameter
% y the labels 
% The inner products between the training and t test points 
% are stored in the matrix Ktest of dimension ell x t
% the true test labels are stored in ytruetest

lambda=gamma;

ell = size(K,1);
ellplus = (sum(y) + ell)/2;
yplus = 0.5*(y + 1);
ellminus = ell - ellplus;
yminus = yplus - y;
rescale = ones(ell,1)+y*((ellminus-ellplus)/ell);
plusfactor = 2*ellminus/(ell*ellplus);
minusfactor = 2*ellplus/(ell*ellminus);
B = diag(rescale) - (plusfactor * yplus) * yplus' ...
      - (minusfactor * yminus) * yminus';
alpha = (B*K + lambda*eye(ell,ell))\y;
b = 0.25*(alpha'*K*rescale)/(ellplus*ellminus);


if length(varargin)==1
    Ktest=varargin{1};
    t = size(Ktest,2);
    ytest = sign(Ktest'*alpha - b);
    varargout{1}=ytest;
elseif length(varargin)==2
    Ktest=varargin{1};
    ytruetest=varargin{2};
    t = size(Ktest,2);
    ytest = sign(Ktest'*alpha - b);
    error = sum(abs(ytruetest - ytest))/(2*t);
    varargout{1}=ytest;
    varargout{2} = error;
end
