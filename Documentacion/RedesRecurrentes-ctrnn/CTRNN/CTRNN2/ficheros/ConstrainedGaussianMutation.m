function mutationChildren = ConstrainedGaussianMutation(parents,options,GenomeLength,FitnessFcn,state,thisScore,thisPopulation) %abel ,scale,shrink

% set boundary constraints (this part changed when there were more parameters being evolved)

%2 o 13 segun corresponda
biases = floor(length(thisPopulation)/3.2);
LB = ones(1,length(thisPopulation))*-6; LB(end-biases+1:end) = ones(1,biases)*-10;%abel [-6	-6	-6	-6	-6	-6	-6	-10	-10	-10];
UB = -LB;%abel[ 6	6	6	6	6	6	6	10	10	10];
% set scale of Gaussian scale = 1;
scale = 1; %abel 

C = zeros(1,length(LB));

for i=1:length(parents)

    %Choose old parent
    parent = thisPopulation(parents(i),:);

    %Generate random vector
    A = scale .* randn(1,length(parent)) - scale / 2 + parent;

    %'Reflect' values which are over constraints
    D = A + max(LB - A, C) .* 2 + min(UB - A, C) .* 2;

    %Ensure this doesn't make it go over other constraint boundary 
    mutationChildren(i,:) = min(max(D,LB),UB);

end
