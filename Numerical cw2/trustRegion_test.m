clc;clear all;close all;
% Rosenbrock function
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; 
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; 


%% Parameters 
% Step acceptance relative progress threshold
eta = 0.1;  
maxIter = 100; 
% Stopping tolerance on relative step length between iterations
tol = 1e-6; 


%% Trust region with 2d subspace, $x_0  = (1.2,1.2)^T$
% x0 = [1.2; 1.2]; % near point 
x0 = [-1.2; 1]; %% remote point
% Trust region radius
Delta = 0.2; %[0.2, 1) work well, below many iterations.

[xTR, fTR, nIterTR, infoTR] = trustRegion(F, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter)
error=[];
for j =1:size(infoTR.xs,2)
    errorTemp = norm(infoTR.xs(:,j) - xTR);
    error = [error,errorTemp];
end

figure
plot(infoTR.xind,infoTR.delta)
xlabel('# of iterations')
ylabel('Trust Region Radius')
% title('Radius change using x0=[1.2, 1.2]')
title('Radius change using x0=[-1.2, 1]')



figure
plot(infoTR.xind,error)
xlabel('# of iterations')
ylabel('convergence rate')
% title('Covergence rate using x0= [1.2; 1.2]')
title('Covergence rate using x0= [-1.2; 1]')
