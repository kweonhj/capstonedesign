fclose('all'); clc; clear; close all;
% Define the measured magnetic field (B) at different positions
% B = [Bx1, By1, Bz1; Bx2, By2, Bz2; ...]
B = [1, 0, 0; 0, 2, 0; 0, 0, 3];

% Define the positions of the measurements
% Positions = [x1, y1, z1; x2, y2, z2; ...]
Positions = [1, 0, 0; 0, 1, 0; 0, 0, 1];

% Define the function to minimize (difference between measured B and B(r))
fun = @(m) norm(B - calculateB(Positions, m));

% Initial guess for the magnetic moment vector
initialMagneticMoment = [1e-5, 1e-5, 1e-5];

% Lower and upper bounds for the magnetic moment vector components
lb = [-inf, -inf, -inf];
ub = [inf, inf, inf];

% Call fmincon to minimize the difference
options = optimoptions('fmincon', 'Display', 'iter');
options.OptimalityTolerance = 1e-12;
[magneticMoment, ~] = fmincon(fun, initialMagneticMoment, [], [], [], [], lb, ub, [], options);

% Display the optimized magnetic moment vector
disp('Optimized Magnetic Moment Vector:');
disp(magneticMoment);

% Function to calculate the theoretical magnetic field B(r)
function B = calculateB(positions, magneticMoment)
    mu0 = 4 * pi * 1e-7; % Permeability of free space

    B = zeros(size(positions));
    
    for i = 1:size(positions, 1)
        r = positions(i, :);
        rNorm = norm(r);
        B(i, :) = (mu0 / (4 * pi)) * ((3 * dot(r, magneticMoment) * r - rNorm^2 * magneticMoment) / rNorm^5);
    end
end
