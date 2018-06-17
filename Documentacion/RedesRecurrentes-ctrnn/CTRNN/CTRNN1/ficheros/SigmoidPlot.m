%%   Plots the  response  of  the  Sigmoid  function
%%   b is the  bias
%%   g is the  gain
function  SigmoidPlot (b, g)

    X	= -10:0.1:10; 
    dim = size(X);
    length = dim(1,2);
    S	= zeros(1, length);

    % Calculate  Sigmoid
    for i=1:length
        S(i)  = Sigmoid(g  * (X(i) + b));
    end
    
    %str = sprintf(';g=%g b=%g;',  g,  b);
    str = 'r';
    plot(X, S,  str);  
            
    
    str1 = sprintf('g= %g', g);
    str2 = sprintf('b= %g',b);    
    legend(str1,str2)
end
