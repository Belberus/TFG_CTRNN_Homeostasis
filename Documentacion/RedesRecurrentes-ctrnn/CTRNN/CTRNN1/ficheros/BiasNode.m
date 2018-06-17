
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%   Single   recurrent CTRNN  node with  Bias
%%   x is the  initial value  of  the  node
%%   h is the  time  step  for integration
%%   t is the  length of  simulation 

function BiasNode (x, h,  t)

    NumTimeSteps  = t/h;	%  Num   of  integration  steps 
    tau  = 1;

    for theta = -4:2:4
        X	= zeros(1, NumTimeSteps);
        oldx	= x;	
        X(1)	= oldx;	
        I	= 0;	%%   no input
        w	= 1;	%%   connection is on but  no multiplier

        for TStep = 1:NumTimeSteps
            delta_x = -oldx + (w * Sigmoid(oldx + theta)) + I;
            newx = oldx  + (h  * delta_x); 
            X(TStep+1)  = newx;
            oldx  = newx;
        end

        %  Now   display
        t = 0:NumTimeSteps;
        str = sprintf(';Theta=  %g;', theta);
        %str = 'r';
        %xlabel(’time’), ylabel(’output’),  title(’output vs  time’);
        plot(t, X);
        legend(str);
        hold  on;
    end
    
    hold  off;
end

