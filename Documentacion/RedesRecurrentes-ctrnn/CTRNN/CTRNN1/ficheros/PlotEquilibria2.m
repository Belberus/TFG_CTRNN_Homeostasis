    %   Plots the  equilibrium as a function of  I
    %   where b is the  bias
    %   and w  is the  weight  of  the  connection 
    
function PlotEquilibria2 (w,  bias)
    %   I = y - w  * sigma (y  + bias)
    %   y = f(x)
    X=-20:0.1:20;
    %   plot for bias
    str = sprintf(';w= %g, b=%g;',  w,  bias);
    plot (X,  X  - (w *((1 + exp(-(X + bias))).^(-1)) ), str)
    hold  on
    %   I = y 
    plot(X, X);
    hold off
end
