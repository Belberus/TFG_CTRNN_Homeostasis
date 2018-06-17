%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Standard  logistic  activation function 
function s = Sigmoid  (x)
    s = 1;
    s = s / (1  + exp(-x));
end
