%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Use initial value  of  x=1 and x=-1
%%   h is the  time  step  for integration 0,01
%%   t is the  length of  simulation  5

function SimpleNode (h, t)
NumTimeSteps  = t/h
x = 1;
SimulateNode(x, 1,  h,  t, 'r');
legend()
hold  on;
x = -1;

SimulateNode(x, 1,  h,  t,'g');
%%
str1 = sprintf('x=  %g',1);
str2 = sprintf('x=  %g',-1);
legend(str1,str2) 
%%
hold  off;
end