%   Plots the  equilibrium as a function of  I
%   Will plot for a bias  = -5, 0 and 5
%   W   is the  weight  of  the  connection 

function PlotEquilibriaRange (w)
    %   I = y - w  * sigma (y  + bias) 
    Y=-20:0.1:20;
    %   plot for bias  = -5 
    b  = -5; 
    PlotEquilibria(w,  -5);
    hold  on
    %   plot for bias  = -0
    PlotEquilibria(w, 0);
    %   plot for bias  = 5
    PlotEquilibria(w, 5);
    hold off
end
