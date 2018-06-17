%   Standard  logistic  activation function
%   Sigmoid
function s = SigmoidDerivative (x)
    s = exp(-x);
    d = (1  + exp(-x)) ^  2;
    s = s / d;
end
