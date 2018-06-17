%%   Plots the  response  of  the  derivative  of  the
%%   Sigmoid  function
%%   b is the  bias
%%   g is the  gain

function  SigmoidDerivativePlot(b, g)
    X = -10:0.1:10; 
    dim	= size(X); 
    length = dim(1,2);

    % Calculate change in  sigmoid 
    dS = zeros(1, length);
    %%   Plot derivative
    %hold on;

    for i=1:length
        dS(i) = SigmoidDerivative(g * (X(i) + b));
    end
    
    plot (X,  dS);
end
