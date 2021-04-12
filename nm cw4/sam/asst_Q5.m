clc; clear all; close all;
%% Assignment 4, Question 5
%% Quadratic Penalty (-1, -1)
F.f = @(x) (x(1)-1).^2 + 0.5 * (x(2)-1.5).^2 - 1;
F.df = @(x) [2*(x(1)-1) x(2)-1.5]';
F.d2f = @(x) [2 0
    0 1];
H.f = @(x) x(1).^2 + x(2).^2 - 2;
H.df = @(x) [2*x(1) 2*x(2)]';
H.d2f = @(x) [2 0
    0 2];
x0 = [-1 -1]';
mu = 1; t = 2; tol = 10^(-10); maxIter = 1000;
[xMin, fMin, nIter, infoQP] = Quadratic_Penalty(F, H, x0, mu, t, tol, maxIter);
xMin
fMin

% path
x = linspace(-3,6);
y = linspace(-3,6);
[X,Y] = meshgrid(x,y);
Z = (X-1).^2+0.5*(Y-1.5).^2-1;
contour(X,Y,Z)
hold on
r=sqrt(2);
x00=0;
y00=0;
theta = linspace(0,2*pi,100);
plot(x00 + r*cos(theta),y00 + r*sin(theta),'--','LineWidth',2)
xs_history2 = infoQP.xs;
plot(xs_history2(1,:),xs_history2(2,:),'LineWidth',3)
xlabel('x');ylabel('y')
title('Method: Quadratic Penalty; Starting point: (-1, -1)')


% convergence plot
figure
dx = zeros(1,length(xs_history2));
for i = 1:length(xs_history2)
    dx(i) = norm(xs_history2(:,i)-xs_history2(:,end));
end
plot(0:length(xs_history2)-1, dx, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Quadratic Penalty; Starting point: (-1, -1)')

figure
dx2 = zeros(1,length(xs_history2)-2);
for i = 1:(length(xs_history2)-2)
    dx2(i) = dx(i+1)/dx(i);
end
plot(1:length(xs_history2)-2, dx2, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Quadratic Penalty; Starting point: (-1, -1)')




%% Quadratic Penalty (4, 3)
x0 = [4, 3]';
mu = 1; t = 2; tol = 10^(-10); maxIter = 1000;
[xMin, fMin, nIter, infoQP] = Quadratic_Penalty(F, H, x0, mu, t, tol, maxIter);
xMin
fMin

% path
x = linspace(-3,6);
y = linspace(-3,6);
[X,Y] = meshgrid(x,y);
Z = (X-1).^2+0.5*(Y-1.5).^2-1;
contour(X,Y,Z)
hold on
r=sqrt(2);
x00=0;
y00=0;
theta = linspace(0,2*pi,100);
plot(x00 + r*cos(theta),y00 + r*sin(theta),'--','LineWidth',2)
xs_history2 = infoQP.xs;
plot(xs_history2(1,:),xs_history2(2,:),'LineWidth',3)
xlabel('x');ylabel('y')
title('Method: Quadratic Penalty; Starting point: (4, 3)')


% convergence plot
figure
dx = zeros(1,length(xs_history2));
for i = 1:length(xs_history2)
    dx(i) = norm(xs_history2(:,i)-xs_history2(:,end));
end
plot(0:length(xs_history2)-1, dx, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Quadratic Penalty; Starting point: (4, 3)')

figure
dx2 = zeros(1,length(xs_history2)-2);
for i = 1:(length(xs_history2)-2)
    dx2(i) = dx(i+1)/dx(i);
end
plot(1:length(xs_history2)-2, dx2, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Quadratic Penalty; Starting point: (4, 3)')





%% Augmented Lagrangian (-1, -1)
x0 = [-1 -1]';
mu_0 = 10; nu = 1; tol = 10^(-10); maxIter = 1000;
[xMin, fMin, nIter, infoQP] = Augmented_Lagrangian(F, H, x0, mu_0, nu, tol, maxIter);
xMin
fMin

% path
x = linspace(-3,6);
y = linspace(-3,6);
[X,Y] = meshgrid(x,y);
Z = (X-1).^2+0.5*(Y-1.5).^2-1;
contour(X,Y,Z)
hold on
r=sqrt(2);
x00=0;
y00=0;
theta = linspace(0,2*pi,100);
plot(x00 + r*cos(theta),y00 + r*sin(theta),'--','LineWidth',2)
xs_history2 = infoQP.xs;
plot(xs_history2(1,:),xs_history2(2,:),'LineWidth',3)
xlabel('x');ylabel('y')
title('Method: Augmented Lagrangian; Starting point: (-1, -1)')

% convergence plot
figure
dx = zeros(1,length(xs_history2));
for i = 1:length(xs_history2)
    dx(i) = norm(xs_history2(:,i)-xs_history2(:,end));
end
plot(0:length(xs_history2)-1, dx, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Augmented Lagrangian; Starting point: (-1, -1)')

figure
dx2 = zeros(1,length(xs_history2)-2);
for i = 1:(length(xs_history2)-2)
    dx2(i) = dx(i+1)/dx(i);
end
plot(1:length(xs_history2)-2, dx2, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Augmented Lagrangian; Starting point: (-1, -1)')






%% Augmented Lagrangian (4, 3)
x0 = [4 3]';
mu_0 = 10; nu = 1; tol = 10^(-10); maxIter = 1000;
[xMin, fMin, nIter, infoQP] = Augmented_Lagrangian(F, H, x0, mu_0, nu, tol, maxIter);
xMin
fMin

% path
x = linspace(-3,6);
y = linspace(-3,6);
[X,Y] = meshgrid(x,y);
Z = (X-1).^2+0.5*(Y-1.5).^2-1;
contour(X,Y,Z)
hold on
r=sqrt(2);
x00=0;
y00=0;
theta = linspace(0,2*pi,100);
plot(x00 + r*cos(theta),y00 + r*sin(theta),'--','LineWidth',2)
xs_history2 = infoQP.xs;
plot(xs_history2(1,:),xs_history2(2,:),'LineWidth',3)
xlabel('x');ylabel('y')
title('Method: Augmented Lagrangian; Starting point: (4, 3)')

% convergence plot
figure
dx = zeros(1,length(xs_history2));
for i = 1:length(xs_history2)
    dx(i) = norm(xs_history2(:,i)-xs_history2(:,end));
end
plot(0:length(xs_history2)-1, dx, 'linewidth', 2)
xlabel('Iteration'); ylabel('Rate of Convergence')
set(gca,'FontSize',15)
title('Method: Augmented Lagrangian; Starting point: (4, 3)')

figure
dx2 = zeros(1,length(xs_history2)-2);
for i = 1:(length(xs_history2)-2)
    dx2(i) = dx(i+1)/dx(i);
end
plot(1:length(xs_history2)-2, dx2, 'linewidth', 2)
xlabel('Iteration'); ylabel('Ratio of distance to optimal point')
set(gca,'FontSize',15)
title('Method: Augmented Lagrangian; Starting point: (4, 3)')
