clc;clear all; close all;
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];
functionq7=@(x,y) (x-3*y).^2 + x.^4;

% Starting point
x0 = [10; 10]; 

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.2; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;

% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_BFGS, fLS_BFGS, nIterLS_BFGS, infoLS_BFGS] = descentLineSearch(F, 'bfgs', lsFun, alpha0, x0, tol, maxIter)
error=[];
for j =1:size(infoLS_BFGS.xs,2)
    errorTemp = norm(infoLS_BFGS.xs(:,j) - xLS_BFGS);
    error = [error,errorTemp];
end
figure
plot(error)
xlabel('# of iterations')
ylabel('convergence rate')
title('Covergence rate using x0= [10; 10]')

e=[];
for i=1:(size(infoLS_BFGS.xs,2)-1)
    etemp=norm(eye(2)-infoLS_BFGS.H_k(:,[2*i-1,2*i])*F.d2f(infoLS_BFGS.xs(:,i+1)));
    e=[e,etemp];
end
figure
plot(e)
xlabel('# of iterations')
ylabel('Error')
title('The error of the approximated inverse hessian')
    

x=linspace(-10,15,1000);y=linspace(-5,15,1000);
[X,Y] = meshgrid(x,y);
Z=functionq7(X,Y);
z=functionq7(infoLS_BFGS.xs(1,:),infoLS_BFGS.xs(2,:));
figure()
surf(X,Y,Z,'EdgeColor', 'none')
hold on
plot3(infoLS_BFGS.xs(1,:),infoLS_BFGS.xs(2,:),z,'r')
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title(' Line Search BFGS trajectories')

figure()
plot(infoLS_BFGS.alphas)
xlabel('iterations')
ylabel('step size')
title('BFGS alphas using x0=[10,10]')












