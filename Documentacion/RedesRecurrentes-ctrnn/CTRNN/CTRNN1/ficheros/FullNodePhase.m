%%   Shows  phase portrait for a given  weight  and bias 

function FullNodePhase (w,  bias)

    [T, X]  = meshgrid([0:0.1:2], [-5:0.5:5]);
    dT = ones(size(T));
    %dX  = -X  + (w * ((1-exp(-X)).^-1));
    dX = -X  + (w * ((1 + exp(-(X + bias))).^(-1)) );
    quiver(T,X,dT,dX)
end