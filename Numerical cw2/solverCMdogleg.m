function p = solverCMdogleg(F, x_k, Delta)
% SOLVERCMDOGLEG Solves quadratic constraint trust region problem via 2d subspace
% p = solverCMdogleg(F, x_k, Delta)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% Delta: upper limit on trust region radius
% OUTPUT
% p: step (direction times lenght)
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rul·lan 
    
    pU = -(F.df(x_k)' *F.df(x_k))/(F.df(x_k)'*F.d2f(x_k)*F.df(x_k))*F.df(x_k);
    pB = -inv(F.d2f(x_k))*F.df(x_k);
    
    if norm(pB) <= Delta
        t=2;
    elseif norm(pU) >= Delta
        t=Delta/norm(pU);
    else
        pB_U = pB-pU;
        t= (-pU.'*pB_U+sqrt((pU.'*pB_U)^2-pB_U.'*pB_U*(pU.'*pU-Delta^2)))/(pB_U.'*pB_U);
        t=t+1;
    end
    
    if t<=1
            p = t* pU;
    else
            p = pU + (t-1)*(pB - pU);
    end
end
    
    




