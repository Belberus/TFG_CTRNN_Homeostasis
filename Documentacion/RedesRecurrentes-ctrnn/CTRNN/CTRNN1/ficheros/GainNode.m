%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Single   recurrent CTRNN  node with  Gain
%%   x is the  initial value  of  the  node
%%   h is the  time  step  for integration
%%   t is the  length of  simulation 
function GainNode  (x, h,  t)
    NumTimeSteps  = t/h;	%  Num   of  integration  steps 
    tau  = 1;

    for g = -5:5    
        X	= zeros(1, NumTimeSteps);
        oldx 	= x; X(1) 	= oldx;
        I	= 0; 	%%   no input
        w	= 1; 	%%   connection is on but  no multiplier 
        theta	= 1;

        for TStep = 1:NumTimeSteps
            delta_x = -oldx + (w * Sigmoid(g*(oldx + theta))) + I;
            newx = oldx  + (h  * delta_x); 
            X(TStep+1)  = newx;
            oldx  = newx;
        end

    %  Now   display
    t = 0:NumTimeSteps;
%    str = sprintf(';g=  %g;', g);
    str = 'b';
    %xlabel(’time’), ylabel(’output’),  title(’output vs  time’);
    plot(t, X,  str);
    hold  on;
    end
    
    hold off;
end