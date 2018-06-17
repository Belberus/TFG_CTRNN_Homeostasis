%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WeightMesh
    %   I = y - w  * sigma (y  + bias)
    [y, w]  = meshgrid(-20:20, -20:20);
    %i = y - (w *((exp(-y) * (((1 + exp(-(y))).^(-1))^2) )));
    i = y - (w * ((1 + exp(-(y + 0))).^(-1)));
    xlabel('i');
    ylabel('w');
    mesh(i, w,  y);
end