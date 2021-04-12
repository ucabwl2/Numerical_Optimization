clc;clear all; close all
t = linspace(4/200,4,200); t=t';
x=[3,150,2];
% phi_vec = (3 + 150*t.^2).*exp(-2*t);
phi =@(x) (x(1) + x(2)*t.^2).*exp(-x(3)*t);
phi_vec = phi(x);

sigma = 0.05*max(phi_vec);
% phi_tilde = phi_vec + normrnd(0,sigma,[200,1]);
phi_tilde = phi_vec + sigma.*randn(200,1) + 0;


%% Least Square function
F6.f = @(x) 0.5*sum((phi_tilde - (x(1)+x(2)*t.^2).*exp(-x(3)*t)).^2);
F6.r = @(x) phi_tilde - (x(1)+x(2)*t.^2).*exp(-x(3)*t);
F6.J = @(x) [-exp(-x(3)*t) -(t.^2).*exp(-x(3)*t) (x(1)+x(2)*t.^2).*t.*exp(-x(3)*t)];
F6.df = @(x) F6.J(x)'*F6.r(x);

% inital paramters
x0 = [1 50 1]';
alpha0=1;
maxIter=200;
tol=1e-10;
ls0pts_LS.c1 = 1e-4;
ls0pts_LS.c2 = 0.2;%0.1 for newton, 0.9 for steepest descent, 0.5 compromise
lsFun6 = @(x_k, p_k, alpha0) lineSearch(F6, x_k, p_k, alpha0, ls0pts_LS);

%Gauss Newton
[xMin_GN, fMin_GN, nIter_GN, info_GN] = descentLineSearch2(F6, 'gauss', lsFun6, alpha0, x0, tol, maxIter)

eta = 0.1; Delta = 1; tol = 1e-10;
debug=0;
%Levenberg-Marquardt then use trustRegion_SR1 to calculate 
[xMin_LM, fMin_LM, nIter_LM, info_LM] = trustRegion(F6, x0, @solverCMLM, Delta, eta, tol, maxIter, debug)


% figure
% plot(t,phi_vec,t,phi_tilde,t,phi(xMin_GN),t,phi(xMin_LM))
% xlabel('')
% legend('model','measurement','est_GN','est_LM')

figure
plot(t,phi_tilde,t,phi(xMin_GN))
xlabel('t')
ylabel('f(x,t)')
title('estimated signal using Gaussian Newton vs the measurement')

figure
plot(t,phi_tilde,t,phi(xMin_LM))
xlabel('t')
ylabel('f(x,t)')
title('estimated signal using Levenberg Marquardt vs the measurement')

figure 
plot(t,phi_vec,t,phi(xMin_GN),t,phi(xMin_LM))
xlabel('t')
ylabel('f(x,t)')
title('Both estimated signals vs the ground truth signal')
legend('ground truth','GN','LM')

error_GN=[];
for j =1:size(info_GN.xs,2)
    errorTemp_GN = norm(info_GN.xs(:,j) - xMin_GN);
    error_GN = [error_GN,errorTemp_GN];
end
figure
plot(error_GN)
xlabel('# of iterations')
ylabel('convergence rate')
title('Covergence rate using Gaussian Newton with x0= [1,50,1]')

error_LM=[];
for j =1:size(info_LM.xs,2)
    errorTemp_LM = norm(info_LM.xs(:,j) - xMin_LM);
    error_LM = [error_LM,errorTemp_LM];
end
figure
plot(error_LM)
xlabel('# of iterations')
ylabel('convergence rate')
title('Covergence rate using Levenber Marquardt with x0= [1,50,1]')

