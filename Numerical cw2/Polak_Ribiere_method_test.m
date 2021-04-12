clc;clear all; close all;
% For computation define as function of 1 vector variable
F.f = @(x) x(1)^2 + 5*x(1)^4 + 10*x(2)^2;
F.df = @(x) [2*x(1) + 20*x(1)^3; 20*x(2)];
F.d2f = @(x) [2 + 60*x(1)^2, 0; 0, 20];
functionq7=@(x,y) x.^2 + 5*x.^4 + 10*y.^2;

% Point
% x0 = [10; 10];
x0 = [-5; 7];

% Initialisation
alpha0 = 1;
tol = 1e-12;
maxIter = 100;

lsOptsCG_LS.c1 = 1e-4;
lsOptsCG_LS.c2 = 0.1;
lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOptsCG_LS);
[xCG_PR_LS, fCG_PR_LS, nIterCG_PR_LS, infoCG_PR_LS] = nonlinearConjugateGradient(F, lsFun, 'PR', alpha0, x0, tol, maxIter)
error=[];
for j =1:size(infoCG_PR_LS.xs,2)
    errorTemp = norm(infoCG_PR_LS.xs(:,j) - xCG_PR_LS);
    error = [error,errorTemp];
end
figure
plot(0:length(error)-1,error)
xlabel('# of iterations')
ylabel('convergence rate')
% title('Covergence rate using x0= [10; 10]')
title('Covergence rate using x0= [-5; 7]')


x=linspace(-10,15,1000);y=linspace(-5,15,1000);
[X,Y] = meshgrid(x,y);
Z=functionq7(X,Y);
z=functionq7(infoCG_PR_LS.xs(1,:),infoCG_PR_LS.xs(2,:));
figure()
surf(X,Y,Z,'EdgeColor', 'none')
hold on
plot3(infoCG_PR_LS.xs(1,:),infoCG_PR_LS.xs(2,:),z,'r')
xlabel('x')
ylabel('y')
zlabel('f(x,y)')
title(' Fletcher-Reeves trajectories')