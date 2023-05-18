fclose('all'); clc; clear; close all;

% Define the measured magnetic field (B) at different positions
% B = [Bx1, By1, Bz1; Bx2, By2, Bz2; ...]
% Define the positions of the measurements
% Positions = [x1, y1, z1; x2, y2, z2; ...]

Bx = readmatrix('ax.txt');
By = readmatrix('ay.txt');
Bz = readmatrix('az.txt');
B = zeros(size(Bx,1), 3);
Positions = zeros(size(Bx,1), 3);

for i = 1:size(B,1)
    B(i, :) = [Bx(i,4), By(i,4), Bz(i,4)];
    Positions(i,1:3) = Bx(i,1:3).*1e-3;
end

% Define the initial guess for the coil centroid position and magnetic moment
initialGuess = [1, 1, 0, 1e-3, 1e-3, 1e-3]; % [x, y, z, mx, my, mz]

% Lower and upper bounds for the optimization variables
lb = [-inf, -inf, -inf, -inf, -inf, -inf];
ub = [inf, inf, inf, inf, inf, inf];

% Call lsqnonlin to perform the nonlinear least squares optimization
options = optimoptions('lsqnonlin', 'Display', 'iter');
options.FunctionTolerance = 1e-12;
options.StepTolerance = 1e-12;
[optVariables, ~] = lsqnonlin(@(variables) objectiveFunction(variables, B, Positions), initialGuess, lb, ub, options);

% Extract the optimized variables
optCentroid = optVariables(1:3);
optMagneticMoment = optVariables(4:6);

% Display the optimized results
disp('Optimized Centroid Position:');
disp(optCentroid);
disp('Optimized Magnetic Moment Vector:');
disp(optMagneticMoment);

% Objective function: Minimize the difference between measured B and B(r)
function error = objectiveFunction(variables, B, Positions)
    centroid = variables(1:3);
    magneticMoment = variables(4:6);
    B_predicted = calculateB(Positions, centroid, magneticMoment);
    error = B - B_predicted;
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
