clc; clear all; close all;
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];
functionq7=@(x,y) (x-3*y).^2 + x.^4;
                     
% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations
debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

% Starting point
x0 = [10; 10]; 

% Trust region parameters 
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 1; % Trust region radius

% Minimisation with 2d subspace and dogleg trust region methods
Fsr1 = rmfield(F,'d2f');
[xTR_SR1, fTR_SR1, nIterTR_SR1, infoTR_SR1] = trustRegion(Fsr1, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, debug)
error=[];
for j =1:size(infoTR_SR1.xs,2)
    errorTemp = norm(infoTR_SR1.xs(:,j) - xTR_SR1);
    error = [error,errorTemp];
end



figure
plot(infoTR_SR1.xind,error)
xlabel('# of iterations')
ylabel('Trust Region Radius')
title('Convergence rate using x0=[10, 10]')

e=[];
for i=1:(size(infoTR_SR1.xs,2)-1)
    etemp=norm(eye(2)-infoTR_SR1.B_k(:,[2*i-1,2*i])*F.d2f(infoTR_SR1.xs(:,i+1)));
    e=[e,etemp];
end
figure
plot(e)
xlabel('# of iterations')
ylabel('Error')
title('The error of the approximated hessian')

x=linspace(-10,15,1000);y=linspace(-5,15,1000);
[X,Y] = meshgrid(x,y);
Z=functionq7(X,Y);
z=functionq7(infoTR_SR1.xs(1,:),infoTR_SR1.xs(2,:));
figure()
surf(X,Y,Z,'EdgeColor', 'none')
hold on
plot3(infoTR_SR1.xs(1,:),infoTR_SR1.xs(2,:),z,'r')
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title(' Trust Region SR1 trajectories')

figure
% plot(infoTR_SR1.xind,infoTR_SR1.delta)
plot(infoTR_SR1.Deltas)
xlabel('# of iterations')
ylabel('Trust Region Radius')
% title('Radius change using x0=[1.2, 1.2]')
title('Radius change using x0=[10, 10]')