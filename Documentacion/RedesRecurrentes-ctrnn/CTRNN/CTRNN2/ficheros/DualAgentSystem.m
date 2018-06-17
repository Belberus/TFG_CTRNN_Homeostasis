function DualAgentSystem(input)

% translate inputs 
WIn(1,:) = input(1:7);
WIn(2,:) = input(8:14);
WIn(3,:) = input(15:21);
WOut(1,:) = input(22:26);
WOut(2,:) = input(27:31);
g_sensors = input(32);
bias_sensors = input(33);
g_interneurons = input(34:38);
bias_interneurons = input(39:43); 
bias_motors(1) = input(44); 
bias_motors(2) = input(45);

y_agent = 0;
X_Start = 160:2:240; %120:1:280; %abel 160:2:240
L = length(X_Start);

%set parameters 
h=0.1;
tau = 1; 
dia_obj = 30; 
dia_agent = 30; 
gain_motors = 1;

%Bilateral symmetry 
for a = 1:7
    WIn(5,:) = WIn(1,8-a); WIn(4,:) = WIn(2,8-a);
end

%Initialise Sensor Neurons
sensor_one = zeros(7,1); 
sensor_two = zeros(7,1); 
interNeuron_one = zeros(5,1); 
interNeuron_two = zeros(5,1); 
motorNeuronA_one=0; 
motorNeuronB_one=0; 
motorNeuronA_two=0; 
motorNeuronB_two=0;

y_obj_velocity = -2.5;

%abel
y_obj = 275;
position = zeros(2,y_obj-y_agent-dia_obj/2,L);
%/abel

    for TrialNumber = 1:L
        %initialise locations/velocities 
        x_obj = X_Start(TrialNumber); 
        y_obj = 275;
        x_agent = 200;

        elapsedTime = 0 - h;
        g = 0;
        while (y_obj - dia_obj/2) > 0 
            g = g + 1;
            elapsedTime = elapsedTime + h;

            % Get Sensor values
            I_one = sensor_intensity(x_obj, y_obj, x_agent, y_agent, dia_obj, dia_agent,1);
            I_two_temp = sensor_intensity(-x_agent, -y_agent, -x_obj, -y_obj, dia_agent, dia_obj,1);
            for f = 1:7
                I_two(f) = I_two_temp(8-f);
            end
            
            %Update Sensor Neurons 
            for a =1:7
                sensor_one(a) = euler_sensor(sensor_one(a),h,tau,I_one(a));
                sensor_two(a) = euler_sensor(sensor_two(a),h,tau,I_two(a)); 
            end

            %update inter Neurons 
            for i = 1:5
                interNeuron_one(i) = euler_IN(interNeuron_one(i),sensor_one,h,tau,WIn(i,:),bias_sensors,g_sensors);
                interNeuron_two(i) = euler_IN(interNeuron_two(i),sensor_two,h,tau,WIn(i,:),bias_sensors,g_sensors);
            end

            %Update motor Neurons 
            motorNeuronA_one = euler_motor(motorNeuronA_one,interNeuron_one,h,tau,WOut(1,:),bias_interneurons,g_interneurons);
            motorNeuronB_one = euler_motor(motorNeuronB_one,interNeuron_one,h,tau,WOut(2,:),bias_interneurons,g_interneurons);
            motorNeuronA_two = euler_motor(motorNeuronA_two,interNeuron_two,h,tau,WOut(1,:),bias_interneurons,g_interneurons);
            motorNeuronB_two = euler_motor(motorNeuronB_two,interNeuron_two,h,tau,WOut(2,:),bias_interneurons,g_interneurons);

            %Update motors
            motorA_one = sigma(motorNeuronA_one,bias_motors(1),gain_motors); motorB_one = sigma(motorNeuronB_one,bias_motors(2),gain_motors); motorA_two = sigma(motorNeuronA_two,bias_motors(1),gain_motors); motorB_two = sigma(motorNeuronB_two,bias_motors(2),gain_motors);

            %Move Agent
            x_agent_velocity = (motorA_one - motorB_one) * 5;
            x_agent = x_agent + h * x_agent_velocity;

            %Move Object
            x_obj_velocity = (motorA_two - motorB_two) * 5;
            x_obj = x_obj + h * x_obj_velocity;
            y_obj = y_obj + h * y_obj_velocity;

            %Keep agent within bounds of environment 
            x_agent = max(0,x_agent);
            x_agent = min(400,x_agent);

            %Keep object within bounds of environment 
            x_obj = max(0,x_obj);
            x_obj = min(400,x_obj);
            elapsedTimePlot(g) = elapsedTime;
            separation(g,TrialNumber) = x_obj - x_agent;
            
            position(:,g,TrialNumber) = [x_obj x_agent];

        end
    end

figure(1)
plot(separation,'r') 
view([-90 -90])
figure(2)
hold on
plot(squeeze(position(1,:,:)),'b')
plot(squeeze(position(2,:,:)),'r')
view([-90 -90])
hold off
% hold on;
% for r = 1:L
%     % vertPosObject --> Separacion Vertical
%     vertPosObject(r,:) = 250 - r;
%     
%     
%     plot(separation(r,:),vertPosObject(r,:),'b');
% end
% 
% hold off;

function [sensor_intensity] = sensor_intensity(x_obj, y_obj, x_agent, y_agent, dia_obj, dia_agent, obj_type)

rad_obj = dia_obj/2;
rad_adj = dia_agent/2;
Y = y_obj - y_agent - rad_obj;

cosTheta(1) = 0.9659; 
cosTheta(2) = 0.9914; 
cosTheta(3) = 0.9962; 
cosTheta(4) = 1; 
cosTheta(5) = 0.9962; 
cosTheta(6) = 0.9914; 
cosTheta(7) = 0.9659;
tanTheta(1) = 0.2679; 
tanTheta(2) = 0.1317; 
tanTheta(3) = 0.0875; 
tanTheta(4) = 0; 
tanTheta(5) = -0.0875;
tanTheta(6) = -0.1317; 
tanTheta(7) = -0.2679;

if obj_type == 0
    %object is a circle
    X1 = x_agent - x_obj - rad_obj * 0.7071; 
    X2 = x_agent - x_obj + rad_obj * 0.7071;

    funcSinTanTheta(1) = 0.1986; 
    funcSinTanTheta(2) = 0.1145; 
    funcSinTanTheta(3) = 0.0799; 
    funcSinTanTheta(4) = 0; 
    funcSinTanTheta(5) = -0.0951; 
    funcSinTanTheta(6) = -0.1488; 
    funcSinTanTheta(7) = -0.3373;

    for a = 1:7
    %if (tan(theta(a)) > X1/Y) & (tan(theta(a)) < X2/Y)
        if (tanTheta(a) > X1/Y) & (tanTheta(a) < X2/Y)
        %can see object
        %Distance = (Y / cos(theta(a))) + (dia_obj * (1-sin(theta(a))) * tan(theta(a))) - dia_agent/2;
        Distance = (Y / cosTheta(a)) + (dia_obj * funcSinTanTheta(a)) - rad_adj;
            if Distance <= 0
                %object touching or 'inside' agent 
                sensor_intensity(a) = 10;
            else if Distance < 221
                sensor_intensity(a) = min(10, 10/Distance);
                else
                    %object too far away 
                    sensor_intensity(a) =0;
                end
            end
        else
            %can't see object 
            sensor_intensity(a) = 0;
        end
    end
else 
	%object is a line
    X1 = x_obj - x_agent + rad_obj; 
    X2 = x_obj - x_agent - rad_obj;

    for a = 1:7
        if (tanTheta(a) > X1/Y) & (tanTheta(a) < X2/Y) 
            Distance = (Y / cosTheta(a));
            if Distance <= 0
                %object touching or 'inside' agent 
                sensor_intensity(a) = 10;
            else if Distance < 221
                sensor_intensity(a) = min(10, 10/Distance);
                else
                    %object too far away 
                    sensor_intensity(a) =0;
                end
            end
        else
            %can't see object 
            sensor_intensity(a) = 0;
        end
    end
end

function [y_new_sensor] = euler_sensor(y_old_sensor, h, tau, I)
y_new_sensor = y_old_sensor + (h/tau) * (I - y_old_sensor);

function [y_new_motor] = euler_motor(y_old, interNeuron, h, tau, w_row, bias,gain) 
W = 0;
    for a = 1:5
        W = w_row(a) * sigma(interNeuron(a),bias(a),gain(a)) + W;
    end

    y_new_motor = y_old + (h/tau) * (-y_old + W );

function [y_new_IN] = euler_IN(y_old, sensor, h, tau, w_row, bias,gain) 
W = 0;
    for a = 1:7
        W = w_row(a) * sigma(sensor(a),bias,gain) + W;
    end
    y_new_IN = y_old + (h/tau) * (-y_old + W );

function [sigma] = sigma(y,bias,gain)
sigma = 1 / (1+exp( (- y - bias)*gain) );
 

