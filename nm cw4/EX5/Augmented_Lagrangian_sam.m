function [xMin, fMin, nIter, infoAL] = Augmented_Lagrangian(F, H, x0, mu, nu, tol, maxIter)
% INPUTS
% F: structure with fields
%  - f: function to minimise
%  - df: gradient of function 
%  - d2f: Hessian of function 
% x0: initial iterate
% mu: initial mu, which will be kept constant
% nu: initial nu, which will be updated in each iteration
% tol: tolerance on two consecutive solutions x_t and x_{t+1}
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% infoAL: structure with information about the iteration 
%   - xs: iterate history for x 
%   - fs: iterate history for f(x)

% Initialize
nIter = 0;
stopCond = false;
x_k = x0;
infoAL.xs = x_k; infoAL.fs = F.f(x_k);

% Parameters
alpha0 = 1; 
opts.c1 = 1e-4; opts.c2 = 0.9; opts.rho = 0.5;
tolNewton = 1e-12;
maxIterNewton = 100;

% Iterate
while (~stopCond && nIter < maxIter)
    if mod(nIter,10) == 0 && nIter>0
        disp(strcat('Iteration ', int2str(nIter)));
    end
    % Create function handler for Q (needs to be redifined at each step because of changing "t")
    G.f = @(x) F.f(x) + nu * H.f(x) + mu / 2 * (H.f(x))^2;
    G.df = @(x) F.df(x) + (nu + mu .* H.f(x)) .* H.df(x);
    G.d2f = @(x) F.d2f(x) + nu .* H.d2f(x) + mu * H.f(x) .* H.d2f(x) + mu * H.df(x) * (H.df(x))';
    lsFun = @(x_k, p_k, alpha0) backtracking(G, x_k, p_k, alpha0, opts);
    
    x_k_1 = x_k;
    [x_k, f_k, ~, ~] = descentLineSearch(G, 'newton', lsFun, alpha0, x_k, tolNewton, maxIterNewton);   
    
    % Check stopping condition
    if norm(x_k - x_k_1) < tol; stopCond = true; end
    
    % Update nu
    nu = nu + mu * H.f(x_k);
    
    % Store info
    infoAL.xs = [infoAL.xs x_k];
    infoAL.fs = [infoAL.fs f_k];
    
    % Increment number of iterations
    nIter = nIter + 1;
end

% Assign values
xMin = x_k;
fMin = F.f(x_k);
end