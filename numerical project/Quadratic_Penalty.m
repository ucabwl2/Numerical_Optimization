function [xMin, fMin, t, nIter, infoQP] = Quadratic_Penalty(x0, mu, t, tol, maxIter,N,Aeq,lb,ub,H,f)
% INTERIORPOINT_BARRIER function to minimise a quadratic form with constraints
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% x0: initial iterate
% mu: initial mu
% t: increasing factor for mu
% tol: tolerance on two consecutive solutions x_t and x_{t+1}
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoBarrier: structure with information about the iteration 
%   - xs: iterate history for x

% Initilize
nIter = 0;stopCond = false;x_k = x0;infoQP.xs = x_k;infoQP.fs = [];
% Parameters for centering step
alpha0 = 1; opts.c1 = 1e-4;opts.c2 = 0.9;opts.rho = 0.5;tolNewton = 1e-12;maxIterNewton = 100;
% Loop 
while (~stopCond && nIter < maxIter)
%     disp(size(Aeq))
%     disp(size(x_k))
    % Create function handler for Q (needs to be redifined at each step because of changing "t")
G.f =@(x) 0.5*x'*H*x + f'*x + (mu/2).*sum((max(lb-x,0)).^2)+ (mu/2).*sum((max(x-ub,0)).^2)+ (mu/2)*sum((Aeq*x).^2);
G.df =@(x) H*x+f + mu*((Aeq*x)*Aeq') + mu*penalty_grad(x,N,lb,ub);
G.d2f =@(x) H + mu*(Aeq'*Aeq);
      
    lsFun = @(x_k, p_k, alpha0) backtracking(G, x_k, p_k, alpha0, opts);
    x_k_1 = x_k;
    [x_k, f_k, nIterLS, infoIter] = descentLineSearch(G, 'newton', lsFun, alpha0, x_k, tolNewton, maxIterNewton);   
    % Check stopping condition
    if norm(x_k - x_k_1) < tol; stopCond = true; end
    % Increase mu
    mu = mu*t;
    % Store info
    infoQP.xs = [infoQP.xs x_k];
    infoQP.fs = [infoQP.fs f_k];
    % Increment number of iterations
    nIter = nIter + 1;
end
% Assign values
xMin = x_k;
fMin = G.f(x_k);