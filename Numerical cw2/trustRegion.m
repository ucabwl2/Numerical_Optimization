function [x_k, f_k, k, info] = trustRegion(F, x0, solverCM, Delta, eta, tol, maxIter)
% TRUSTREGION Trust region iteration
% [x_k, f_k, k, info] = trustRegion(F, x0, solverCM, Delta, eta, tol, maxIter)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% solverCM: handle to solver to quadratic constraint trust region problem
% Delta: upper limit on trust region radius
% eta: step acceptance relative progress threshold
% tol: stopping condition on minimal allowed step
%      norm(x_k - x_k_1)/norm(x_k) < tol;
% maxIter: maximum number of iterations
% OUTPUT
% x_k: minimum
% f_k: objective function value at minimum
% k: number of iterations
% info: structure containing iteration history
%   - xs: taken steps
%   - xind: iterations at which steps were taken
%   - stopCond: shows if stopping criterium was satisfied, otherwsise k = maxIter
%   
% Reference: Algorithm 4.1 in Nocedal Wright
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan 
niter=0;
x_k_1=x0;
info.xs=x0;
info.xind=0;
info.stopCond=false;
info.delta=0;
    for i=1:maxIter
        niter=niter+1;
        x_k=x_k_1;
        p = solverCM(F, x_k, Delta);
%         rho_k=solverCM(F,x_k,Delta);
         rho_k=(F.f(x_k)-F.f(x_k+p))/(m(x_k,zeros(2,1))-m(x_k,p));

        if rho_k<0.25
            Delta_new =0.25*Delta;
        else
            if rho_k > 0.75 && sqrt(abs(p'*p))== Delta
                Delta_new=min(2*Delta,10000);
            else
                Delta_new=Delta;
            end
        end
        Delta=Delta_new;

        if rho_k>eta
            x_k_1=x_k+p;
        else
            x_k_1=x_k;
        end
        
        if norm(x_k - x_k_1)/norm(x_k) < tol
            if x_k==x_k_1
                continue
            else
            info.stopCond=[info.stopCond,true]
            break
            end
        end
        info.xs=[info.xs,x_k_1];
        info.xind=[info.xind,niter];
        info.delta=[info.delta,Delta];
    end
    k=niter;
    x_k=x_k_1;
    f_k=F.f(x_k);

     function y=m(x,p)
         y=F.f(x_k)+F.df(x_k)'*p+0.5*p'*F.d2f(x_k)*p;
     end

end