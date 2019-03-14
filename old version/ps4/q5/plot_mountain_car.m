function plot_mountain_car(x)

clf; hold on;
plot(-1.2:0.1:0.5, 0.3*sin(3*(-1.2:0.1:0.5)), 'k-');

theta = atan2(3*0.3*cos(3*x(1)), 1.0);
y = 0.3*sin(3*x(1));
car = [-0.05 0.05; 0.05 0.05; 0.05 0.01; -0.05 0.01; -0.05 0.05]';
fwheel = [0.035 + 0.01*cos([0:0.5:2*pi 0]); 0.01 + 0.01*sin([0:0.5:2*pi 0])];
rwheel = [-0.035 + 0.01*cos([0:0.5:2*pi 0]); 0.01 + 0.01*sin([0:0.5:2*pi 0])];

R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
car = (R*car + repmat([x(1); y], 1, size(car,2)));
fwheel = (R*fwheel + repmat([x(1); y], 1, size(fwheel,2)));
rwheel = (R*rwheel + repmat([x(1); y], 1, size(rwheel,2)));

plot(car(1,:), car(2,:));
plot(fwheel(1,:), fwheel(2,:));
plot(rwheel(1,:), rwheel(2,:));

axis([-1.3 0.6 -0.4 0.4]);
axis equal;

