%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Minimal  CTRNN3
%%   Shows  the  effect of  varying Tau and input for a CTRNN  Node
%%   h is the  time  step  for integration
%%   t is the  length of  simulation 
function MinimalCTRNN3  (x, h,  t)
    NumTimeSteps  = t/h;	%  Num   of  integration  steps
    HalfTimeStep  = 2/h;	%   halfway  point 
    tau  = [0.5, 1,  2];
    I	= [-4,  4];
    t = 0:NumTimeSteps;

    for i=1:3
        for j=1:2
            X	= zeros(1, NumTimeSteps);
            oldx 	= x; X(1) 	=  oldx; 
            tau_value	= 1 /  tau(i);
            
            for TStep = 1:NumTimeSteps
                if (TStep  < HalfTimeStep)
                    delta_x = tau_value * (-oldx  + I(j)); 
                else
                    delta_x = tau_value * (-oldx);	
                end
                newx = oldx  + (h  * delta_x); 
                X(TStep+1)  = newx;
                oldx  = newx;
            end
 
            %  Now   display
            %str = sprintf(';I= %g, T=%g;',  I(j), tau(i))
            str = 'b'; % color
            %xlabel('time'), ylabel('output'),  title('output vs  time');
            plot(t, X,  str);
            hold  on
        end
    end
    hold  off;
end
