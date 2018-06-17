%%%%%%%%%%%%%%%
%%% Debo estar ubicado en la carpeta que contiene el ConstrainedGaussian y el OrientationEvolve
%% Invocar en la consola de matlab como ---> GeneticAlgorithm1

%%% limpio la consola y las variables del sistema %%%%
clear
clc


%%%----------- Inicializo las Variables ---------%%%%
%%% Funciones %%%

% Funcion de seleccion
selectionFunction =@selectionroulette

% Funcion fitness scaling
fitnessScalingFunction = @fitscalingprop


%% Funcion de mutacion 
mutationFunction = @ConstrainedGaussianMutation
%mutationFunction = @mutationgaussian

%%---- Escoger una de las dos y Comentar la otra ----%%%%%
%% 1.Para OrientationEvolve 
FitnessFunction = @OrientationEvolve;
populationSize = 10
numberOfVariables = 10
initialPopulation = [-6, -6, -6, -6, -6, -6, -6, -10, -10, -10]
initialScores = [6, 6, 6, 6, 6, 6, 6, 10, 10, 10]

%% 2.Para ObjectDisciminateInterNeuronsEvolv
% FitnessFunction = @ObjectDiscriminateInterNeuronsEvolve
% populationSize = 45 
% numberOfVariables = 45
% initialPopulation = 1:45 %% Arreglo de 1 a 45
% initialScores = 1:45

%%%%---------------------------------------------------%%%%%%%%

%%% ---- Generaciones ---%%%
generations = 200

%%% ---- graficas a mostrar ----%%
plots = {@gaplotbestf,@gaplotstopping,@gaplotselection}  


%%%% --------- Parametrizo la función ----------- %%%%%%%%%%
opts = gaoptimset('PlotFcns',plots,...
'PopulationSize',populationSize,...
'InitialPopulation',initialPopulation,...
'InitialScores',initialScores,....
'Generations', generations,....
'MutationFcn', mutationFunction,...
'SelectionFcn',selectionFunction,...
'FitnessScalingFcn',fitnessScalingFunction,...
'Display','iter',... %%% Con Display iter puedo ver la informacion de cada iteracion por consola
'PlotInterval',1)


%%% --------------- Ejecuto el GA -------------------------------%%%
[x,Fval,exitFlag,Output] = ga(FitnessFunction,numberOfVariables,[],[],[],[],[],[],[],opts);

%%%%% ------ Salida del GA ----- %%%%%
fprintf('The number of generations was : %d\n', Output.generations);
fprintf('The number of function evaluations was : %d\n', Output.funccount);
fprintf('The best function value found was : %g\n', Fval);
