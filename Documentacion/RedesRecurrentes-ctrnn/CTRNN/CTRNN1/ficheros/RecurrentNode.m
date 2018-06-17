%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%   Minimal  CTRNN  node with  recurrent  connection
%%   x is the  initial value
%%   h is the  time  step  for integration
%%   t is the  length of  simulation 
function RecurrentNode (x, h,  t)
    NumTimeSteps  = t/h;	%  Num   of  integration  steps 
    tau  = 1;

    for w  = -4:2:4
        X	= zeros(1, NumTimeSteps);	
        oldx	= x;	
        X(1)	= oldx;	
        I	= 0;	%  No   input
        theta	= 0;	%  No   bias

        for TStep = 1:NumTimeSteps
            delta_x = -oldx + (w * Sigmoid(oldx + theta)) + I;
            newx = oldx  + (h  * delta_x); 
            X(TStep+1)  = newx;
            oldx  = newx;
        end
        
        %  Now   display
        t = 0:NumTimeSteps;
        %str = sprintf(';W=  %g;', w);
        str = 'r';
        %xlabel('time'), ylabel('output'),  title('output vs  time');
        plot(t, X,  str);
        hold  on;
    end
    hold off;
end
