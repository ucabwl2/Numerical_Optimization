%June11,2016
%SVM
function [alpha,Ker,beta0]=SVM_mine(X,Y,kernel)
% X is N*p, Y is N*1,{-1,1}
% Constant=Inf for Hard Margin
global  precision Cost

switch kernel
    case 'linear'
        Ker=Ker_Linear(X,X);
    case 'ploynomial'
        Ker=Ker_Polynomial(X,X);
    case 'RBF'
        Ker=Ker_RBF(X,X);
    case 'Sigmoid'
        Ker=Ker_Sigmoid(X,X);
end

N= size(X,1);
% disp(size(diag(Y)))
% disp(size(Ker))
H= diag(Y)*Ker*diag(Y);
f= - ones(N,1);
Aeq=Y';
lb = zeros(N,1);
ub = repmat(Cost,N,1);
alpha0=0.01*ones(N,1);
mu = 1; t = 2; tol = 10^(-10); maxIter = 1000;


% [alpha, fMin, nIter, infoQP] = Quadratic_Penalty(alpha0, mu, t, tol, maxIter,N,Aeq,lb,ub,H,f);
A=[];
b=[];
beq=0;
% alpha=quadprog(H,f,A,b,Aeq,beq, lb, ub);

%%%My extra code

F.f =@(x) 0.5*x'*H*x + f'*x;
F.df =@(x) H*x+f; 
F.d2f =@(x) H;

phi.f = @(x) sum(log(x)) - sum(log( Cost * ones(size(x)) - x));
phi.df = @(x) -1 * (ones(size(x))./x + ones(size(x))./ (x - Cost * ones(size(x))));
phi.d2f = @(x)  diag(ones(size(x)) ./ (x.^2) ) + ( diag(ones(size(x)) ./ (x- Cost * ones(size(x)).^2)));

[alpha, fMin, t, nIter, infoBarrier] = interiorPoint_Barrier(F, phi, alpha0, t, mu, tol, maxIter);


serial_num=(1:size(X,1))';
serial_sv=serial_num(alpha>precision&alpha<Cost);

temp_beta0=0;
for i=1:size(serial_sv,1)
    temp_beta0=temp_beta0+Y(serial_sv(i));
    temp_beta0=temp_beta0-sum(alpha(serial_sv(i))*...
        Y(serial_sv(i))*Ker(serial_sv,serial_sv(i)));
end
beta0=temp_beta0/size(serial_sv,1);

return