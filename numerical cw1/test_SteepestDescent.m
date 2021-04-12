clear all;
clc;close all;
% For computation define as function of 1 vector variable
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; % function handler, 2-dim vector
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  % gradient handler, 2-dim vector
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; % hessian handler, 2-dim vector
% For visualisation proposes define as function of 2 variables
rosenbrock = @(x,y) 100.*(y - x.^2).^2 + (1 - x).^2;

% rosenbrock.df_dx=@(x,y) (y -x.^2).*x-2.*(1-x);
% rosenbrock.df_dy=@(x,y) 200*(y-x.^2);
% 
% rosenbrock.ddf_ddx=@(x,y) -400*(y-3*x.^2)+2;
% rosenbrock.ddf_dxdy=@(x,y) -400*x;
% rosenbrock.ddf_ddy=@(x,y) 200;
%%

% Initialisation
alpha0 = 1;
maxIter = 1e4;
alpha_max = alpha0;
tol = 1e-6;
%=============================
% Point x0 = [1.2; 1.2]
%=============================
% x0 = [1.2; 1.2]; %excerise 4b
x0 = [-1.2; 1]; %excerise 4c

% Steepest descent line search strong WC
lsOptsSteep.c1 = 1e-4;
lsOptsSteep.rho = 0.1;
lsFun = @(x_k, p_k, alpha0) backtracking(F, x_k, p_k, alpha_max, lsOptsSteep);
[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter)
xMin1 = xSteep(1);
xMin2 = xSteep(2);

figure()
plot(infoSteep.alphas)
xlabel('iterations')
ylabel('step size')
title('Steepest alphas')

x=linspace(-1.5,1.5,1000);y=linspace(-1.5,1.5,1000);
[X,Y] = meshgrid(x,y);
Z=rosenbrock(X,Y);
z=rosenbrock(infoSteep.xs(1,:),infoSteep.xs(2,:));
figure()
surf(X,Y,Z,'EdgeColor', 'none')
hold on
plot3(infoSteep.xs(1,:),infoSteep.xs(2,:),z,'r')
xlabel('x')
ylabel('y')
zlabel('trajectories')
title('Steepest trajectories')


error=[];
for j =1:size(infoSteep.xs,2)
    errorTemp = norm(infoSteep.xs(:,j) - xSteep);
    error = [error,errorTemp];
end
figure
plot(0:length(error)-1,error)
% title('Steepest descent x0= [1.2; 1.2]')
title('Steepest descent x0= [-1.2; 1]')








% assessVariableEqual('xMin1', 1, 'RelativeTolerance', 0.01)
% assessVariableEqual('xMin2', 1, 'RelativeTolerance', 0.01)
