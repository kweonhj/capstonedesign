fclose('all'); clc; clear; close all;
% Define the measured magnetic field (B) at different positions
% B = [Bx1, By1, Bz1; Bx2, By2, Bz2; ...]
B = [1, 0, 0; 0, 2, 0; 0, 0, 3];

% Define the positions of the measurements
% Positions = [x1, y1, z1; x2, y2, z2; ...]
Positions = [1, 0, 0; 0, 1, 0; 0, 0, 1];

% Define the initial guess for the coil centroid position and magnetic moment
initialGuess = [1e-3, 0, 0, 1e-5, 1e-5, 1e-5]; % [x, y, z, mx, my, mz]

% Lower and upper bounds for the optimization variables
lb = [-inf, -inf, -inf, -inf, -inf, -inf];
ub = [inf, inf, inf, inf, inf, inf];

% Call fmincon to minimize the difference
options = optimoptions('fmincon', 'Display', 'iter');
[optVariables, ~] = fmincon(@(variables) objectiveFunction(variables, B, Positions), initialGuess, [], [], [], [], lb, ub, @(variables) constraintFunction(variables, B, Positions), options);

% Extract the optimized variables
optCentroid = optVariables(1:3);
options.OptimalityTolerance = 1e-12;
optMagneticMoment = optVariables(4:6);

% Display the optimized results
disp('Optimized Centroid Position:');
disp(optCentroid);
disp('Optimized Magnetic Moment Vector:');
disp(optMagneticMoment);

% Objective function: Minimize the difference between measured B and B(r)
function cost = objectiveFunction(variables, B, Positions)
    centroid = variables(1:3);
    magneticMoment = variables(4:6);
    B_predicted = calculateB(Positions, centroid, magneticMoment);
    cost = norm(B - B_predicted);
end

% Constraint function: Empty in this case (no constraints)
function [c, ceq] = constraintFunction(~, ~, ~)
    c = [];
    ceq = [];
end

% Function to calculate the theoretical magnetic field B(r)
function B = calculateB(positions, centroid, magneticMoment)
    mu0 = 4 * pi * 1e-7; % Permeability of free space

    B = zeros(size(positions));

    for i = 1:size(positions, 1)
        r = positions(i, :) - centroid;
        rNorm = norm(r);
        B(i, :) = (mu0 / (4 * pi)) * ((3 * dot(r, magneticMoment) * r - rNorm^2 * magneticMoment) / rNorm^5);
    end
end
