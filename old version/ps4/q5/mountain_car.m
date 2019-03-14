function [x_next, s_idx, absorb] = mountain_car(x,a)

% simulate forward
x_next(2) = x(2) + 0.001*a - 0.0025*cos(3*x(1));
x_next(1) = x(1) + x_next(2);

% clip the state to the bounds
absorb = 0;
if (x_next(1) < -1.2) x_next(2) = 0; end
if (x_next(1) > 0.5) absorb = 1; end
x_next(1) = max(min(x_next(1), 0.5), -1.2);
x_next(2) = max(min(x_next(2), 0.07), -0.07);


% find the index of the state
s_idx = 10*floor(10*(x_next(1) + 1.2)/(1.7 + 1e-10)) + ...
  floor(10*(x_next(2) + 0.07)/(0.14 + 1e-10)) + 1;
