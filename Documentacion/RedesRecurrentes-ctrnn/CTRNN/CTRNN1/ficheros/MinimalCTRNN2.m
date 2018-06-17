%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%   Measures the  effect of  t ->  h
%%   h is the  time  step  for integration
%%   t is the  length of  simulation 

function MinimalCTRNN2  (x, h,  t)

    NumTimeSteps  = t/h;
    tau  = h/2;
    X	= zeros(1, NumTimeSteps);
    oldx 	= x; 
    X(1)	= oldx; tau_value	= 1/tau;

    for TStep = 1:NumTimeSteps
        newx = oldx  + (h  * (tau_value * -oldx));
        X(TStep+1)  = newx;
        oldx  = newx;
    end

    %  Now   display
    t = 0:NumTimeSteps;
    max(X);

    %str = sprintf(';T_i= %g;', tau);
    str = 'b';
    xlabel('time'), ylabel('output'), title('output  vs  time');
    plot(t, X,  str);
    hold  on;

    tau  = h;
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
    str = 'r';
    xlabel('time'), ylabel('output'), title('output  vs  time');
    plot(t, X,  str);
    hold  on;
    hold  off;
end
