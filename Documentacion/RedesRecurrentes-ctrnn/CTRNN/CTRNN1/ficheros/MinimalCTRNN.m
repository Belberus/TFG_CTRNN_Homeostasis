%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Minimal  CTRNN  - investigates the  effect of  tau  on the  system
%%   x is the  initial value  of  x  1
%%   h is the  time  step  for integration 0.01
%%   t is the  length of  simulation 5

function MinimalCTRNN  (x, h,  t)
    NumTimeSteps  = t/h;
    
    for tau  = 0.2:0.2:2.0
        X	= zeros(1, NumTimeSteps);
        oldx 	= x; 
        X(1)	= oldx; 
        tau_value	= 1/tau;

        for TStep = 1:NumTimeSteps
            newx = oldx  + (h  * (tau_value * -oldx)); 
            X(TStep+1)  = newx;
            oldx  = newx;
        end
 
        %  Now   display
        t = 0:NumTimeSteps;
        max(X);
        %str = sprintf(';T_i= %g;', tau);
        str = 'r'
        xlabel('time'), ylabel('output'), title('output  vs  time');
        plot(t, X,  str);
        hold  on;
    end
    
    hold  off;
end