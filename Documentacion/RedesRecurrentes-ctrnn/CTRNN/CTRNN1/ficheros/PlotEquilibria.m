%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Plots the  equilibria for %   I = y - w  * sigma (y  + bias)
%  w   is the  weight  of  the  connection
%   b is the  bias
function PlotEquilibria  (w,  b)
    %   I = y - w  * sigma (y  + bias) 
    Y=-20:0.1:20;
    %   plot for weight  and bias 
    str = sprintf(';b=%g;', b); xlabel('I');
    ylabel('y');
    plot (Y  - (w *((1 + exp(-(Y + b))).^(-1)) ), Y,  str);
end
