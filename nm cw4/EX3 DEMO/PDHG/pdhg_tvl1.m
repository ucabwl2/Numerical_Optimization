% Perform TVL1 denoising using adaptive PDHG.  The problem solved is
%            min_u    TV(u) + mu | u - noisy |
%  Inputs...
%    noisy : An 2D array containing the pixels of the noisy image
%    mu    : A scalar, the regularization parameter for the denoising
%    opts  : An optional argument for customizing the behavior of 
%              pdhg_adaptive
%  Outputs...
%    denoised : The denoised image
%    out      : A struct containing convergence data generated by the
%                function 'pdhg_adaptive' when it solved the problem.
% 
%  This File requires that 'pdhg_adaptive.m' be in your current path. 
%
%  For an explanation of PDHG and how it applies to this problem, see
%    "Adaptive Primal-Dual Hybrid Gradient Methods for Saddle-Point
%    Problems"  available at <http://arxiv.org/abs/1305.0546>

function [ denoised ,out ] = pdhg_tvl1( im, mu, opts )

    % Proximal operator for primal variable
    fProx = @(x,tau) x;
    % Proximal operator for dual variables
    gProx = @(y,sigma) [ projectInf(y(1:end/3,:),y(end/3+1:2*end/3,:)) ;...
                                  max(min(mu,y(2*end/3+1:end,:) - sigma*im),-mu)];
    % Linear operators perform the gradient and identity operators
    A = @(x) [Dx(x) ; Dy(x) ; x];
    At = @(y) Dxt(y(1:end/3,:))+Dyt(y(end/3+1:2*end/3,:))+y(2*end/3+1:end,:);
  
    % Initial guess
    [rows,cols] = size(im);
    x0 = zeros(size(im));
    y0 = zeros(3*rows,cols);
   
    %  Options for pdhg_adaptive
    if ~exist('opts','var')
        opts = [];
    end
    %opts.f1 = @(x,y,x0,y0,tau,sigma) sum(sum(sqrt(Dx(x).^2+Dy(x).^2)))+mu*sum(sum(abs(x-im)));

    %  Call the solver
    [denoised ,out]= pdhg_adaptive(x0,y0,A,At,fProx,gProx,opts); 
    

return

function d = Dx(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
d(:,1) = 0;
return

function d = Dxt(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
d(:,cols) = u(:,cols);
d(:,1) = -u(:,2);
return

function d = Dy(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
d(1,:) = 0;
return

function d = Dyt(u)
[rows,cols] = size(u); 
d = zeros(rows,cols);
d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
d(rows,:) = u(rows,:);
d(1,:) = -u(2,:);
return

function rval = projectInf( x,y )
norm = sqrt(max(x.*x+y.*y,1));
x = x./norm;
y = y./norm;
rval = [x;y];
return

