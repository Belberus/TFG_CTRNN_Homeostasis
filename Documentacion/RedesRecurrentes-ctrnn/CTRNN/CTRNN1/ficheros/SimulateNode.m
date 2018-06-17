%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Use euler to  integrate
%%   x is the  initial value
%%   tau  is the  time  constant
%%   h is the  step  size  for integration
%%   t is the  simulation length 

function SimulateNode (x, tau, h,  t,str)
    NumTimeSteps  = t/h;
    X	= zeros(1, NumTimeSteps);
    oldx 	= x ;
    X(1)    = oldx ;
    tau_value	= 1/tau;
    
    for TStep = 1: NumTimeSteps
       newx = oldx  + (h  * (tau_value * -oldx)) ;
        %newx = oldx  + (h  * (tau_value * oldx)) 
        X(TStep+1)  = newx;
        oldx  = newx;
    end

    %  Now   display
    t = 0:NumTimeSteps ;
    
    max(X)
    %str = sprintf('x=  %g',x);
    %str = 'b'; %%%%
    xlabel('time'), ylabel('output'),  title('output vs  time');
    plot(t, X,  str);    
    
end