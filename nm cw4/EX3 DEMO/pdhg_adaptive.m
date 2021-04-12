%%      pdhg_adaptive.m  by Tom Goldstein
%   This method solves the saddle-point problem
%
%    max_y min_x f(x) + <Ax,y> - g(y)
%
%   Using an adaptive PDHG method.  The required inputs are
%   "x" - An Nx1 column vector with the initial guess for x
%   "y" - An Mx1 column vector with the initial guess for y
%   "A" - A matrix or function handle representation of an MxN linear operator.
%   "At"- A matrix or function handle representation of the adjoint of A.
%   "fProx"- A function handle of the form @(x,tau)<do stuff>.
%       This function computes:  min_x  f(x)+(1/2/tau)||x-x0||^2 
%   "gProx"- A function handle of the form @(y,sigma)<do stuff>.
%       This function computes:  min_y  g(y)+(1/2/sigma)||y-y0||^2 
%   "opts" - an optional struct containing various options that the user 
%       may choose to set.  The options are described below in the function
%       "setDefaults" at the bottom of this file.  Common options are
%       opts.maxIters  - the maximum iterations before termination
%       opts.adaptive - set this to 'false' to turn off adaptivity.  If you
%           turn adaptivity off, you must set values for opts.tau and
%           opts.sigma, which are the stepsizes used by PDHG.
%        
%   Note: This code can handle complex matrices provided A and At are
%   Hermitian adjoints.

function [x,outs]= pdhg_adaptive(x, y, A, At, fProx, gProx, opts)

%%  Check whether we have function handles or matrices
if ~isnumeric(A)
    assert(~isnumeric(At),'If A is a function handle, then At must be a handle as well.')
end
%  If we have matrices, create functions so we only have to treat one case
if isnumeric(A)
     At = @(x)A'*x;
     A = @(x) A*x;
end

%%  Check preconditions
% Make sure 'opts' struct exists, and is filled with all the options
if ~exist('opts','var')
    opts = [];
end
opts = setDefaults(opts,x,A,At);
%  Make sure A and At are adjoints of one another
checkAdjoints(A,At,x,y);


%% Get some commonly used values from the 'opts' struct
tau = opts.tau;        % primal stepsize
sigma = opts.sigma;    % dual stepsize
maxIters = opts.maxIters;
a = opts.a;             % adaptivity level
L = opts.L;             % Reciprical spectral radius of A'A
Delta = opts.Delta;     % Used to compare residuals to decide when to update stepsizes

%% Allocate space for the returned variables in the 'outs' struct
outs = [];
outs.tau = zeros(maxIters,1);    % primal stepsizes
outs.sigma = zeros(maxIters,1);  % dual stepsizes
outs.f1 = zeros(maxIters,1);     % optional function evaluation
outs.f2 = zeros(maxIters,1);     % optional function evaluation
outs.p = zeros(maxIters,1);      % primal residuals
outs.d = zeros(maxIters,1);      %  dual residuals

%% Initialize some values
updates = 0;
Ax = A(x);
Aty = At(y);

%maxAx = 0;
%maxAty = 1;
maxPrimal = -Inf;
maxDual = -Inf;

%% Begin Iteration
for iter = 1:maxIters
   
   % store old iterates
   x0 = x;
   y0 = y;
   Ax0=Ax;
   Aty0=Aty;
    
   % primal update
   x = fProx(x-tau*Aty,tau);
   Ax = A(x);
   Axh = 2*Ax-Ax0;
    
   % dual update
   y = gProx(y+sigma*Axh,sigma);
   Aty = At(y);
     
   % compute and store residuals
   dx = x-x0;
   dy = y-y0;
   r1 = dx/tau +Aty0;
   r2 = Aty;
   d1 = dy/sigma +Axh;
   d2 = Ax;
   primal = norm(r1(:)-r2(:));
   dual = norm(d1(:)-d2(:));
   maxPrimal = max(maxPrimal,primal);
   maxDual = max(maxDual,dual);
   
   outs.p(iter) = primal;
   outs.d(iter) = dual;
   
   % store various values that we wish to track
   outs.f1(iter) = opts.f1(x,y);
   outs.f2(iter) = opts.f2(x,y);
   outs.tau(iter) = tau;
   outs.sigma(iter) = sigma;
   
   % Test stopping conditions
   if ( opts.stopNow(x,y,primal,dual,maxPrimal,maxDual) && iter>5) || iter>=maxIters 
     outs.y = y;
     outs.p = outs.p(1:iter);
     outs.d = outs.d(1:iter);
     outs.f1 = outs.f1(1:iter);
     outs.f2 = outs.f2(1:iter);
     outs.updates  = updates;
     outs.tau = outs.tau(1:iter);
     outs.sigma = outs.sigma(1:iter);
     outs.iters = iter;
    return;
   end
     
  % Test the backtracking/stability condition
  dotProduct = (Ax-Ax0).*conj(dy);
  Axy = 2*real(sum(dotProduct(:)));
  Hnorm = norm(dx(:))^2/tau+ norm(dy(:))^2/sigma;
  if opts.backtrack && (opts.gamma*Hnorm<Axy)
      x=x0;
      y=y0;
      Ax=Ax0;
      Aty=Aty0;  
      decay = opts.b*opts.gamma*Hnorm/Axy;
      tau = tau*decay;
      sigma = sigma*decay;
      L = L*decay*decay; 
  end
 
  %  Perform adaptive update
  
  if opts.adaptive && iter>1 && max(primal,dual)< max(outs.p(iter-1),outs.d(iter-1))  
    if  primal>Delta*dual
            tau = tau/(1-a);
            sigma = L/tau;
            a=a*opts.eta;
            updates = updates+1;
    end
    if  primal < dual/Delta
            tau = tau*(1-a);
            sigma = L/tau;
            a=a*opts.eta;
            updates = updates+1;
    end
  end
  

  
end  % end for loop

return % end function


%% Check that A and At represent adjoints
function checkAdjoints(A,At,x,y)
  % Generate random arrays with sizes of x and y
    rx = randn(size(x));
    ry = randn(size(y));
  %  Check that <A(x),y> == <x, At(y)>
    prod1 = A(rx).*conj(ry);
    prod2 = rx.*conj(At(ry));
    dot1 = sum(prod1(:));
    dot2 = sum(prod2(:));
    relativeError = abs(dot1-dot2)/(abs(dot1)+abs(dot2));
    assert(relativeError<1e-6,'At is not the adjoint of A'  );
return


%% Fill in the struct of options with the default values
function opts = setDefaults(opts,x0,A,At)

%  L:  The reciprocal of the spectral radius of A'A.
%  Approximate the spectral radius of A'A if we don't know L
if ~isfield(opts,'L') || opts.L<=0
    x = randn(size(x0));
    transform = At(A(x));
    specRadius = norm(transform,'fro')/norm(x,'fro');
    opts.L = 2/specRadius;
end

%  maxIters: The maximum number of iterations
if ~isfield(opts,'maxIters')
    opts.maxIters = 1000;
end
% tol:  The relative decrease in the residuals before the method stops
if ~isfield(opts,'tol') % Stopping tolerance
    opts.tol = 1e-3;
end
% adaptive:  If 'true' then use adaptive method.
if ~isfield(opts,'adaptive')    %  is Adaptive?
    opts.adaptive = true;
end

% backtrack:  If 'true' then use backtracking method.
if ~isfield(opts,'backtrack')    %  is backtracking?
    opts.backtrack = true;
end

% f1:  An optional function that is computed and stored after every
% iteration
if ~isfield(opts,'f1')          % These functions get evaluated on each iterations, and results are stored
    opts.f1 = @(x,y) 0;
end
% f2:  An optional function that is computed and stored after every
% iteration
if ~isfield(opts,'f2')
    opts.f2 = @(x,y) 0;
end
% tau:  The intial stepsize for the primal variables
if ~isfield(opts,'tau')         % starting value of tau
    opts.tau = sqrt(opts.L);
end
% sigma: The intial stepsize for the dual variables
if ~isfield(opts,'sigma')       % starting value of sigma
    opts.sigma = opts.L/opts.tau;
end

%% Adaptivity parameters
if ~isfield(opts,'a')   %  Intial adaptive update strength for stepsizes
    opts.a = .5; % .5
end
if ~isfield(opts,'eta') %  How fast does the adaptivity level decay
    opts.eta = .95; %.95
end
if ~isfield(opts,'Delta') % update stepsizes when primal/dual ratio exceeds Delta
    opts.Delta = 2; %1.5
end
if ~isfield(opts,'gamma') % Used to determine when need to backtrack to maintain positivity conditions
    opts.gamma = .75;
end
if ~isfield(opts,'b')  % Adaptivity parameter used for backtracking update
    opts.b = .95;
end


%% Stopping conditions

%  The code below is for stopping rules
%  The field 'stopNow' is a function that returns 'true' if the iteration
%  should be terminated.  The field 'stopRule' is a string that allows the
%  user to easily choose default values for 'stopNow'.  The default
%  stopping rule terminates when the relative residual gets small.
if isfield(opts,'stopNow') 
    opts.stopRule = 'custom';
end

% Set the default behavior
if ~isfield(opts,'stopRule') 
    opts.stopRule = 'ratioResidual';
end

if strcmp(opts.stopRule,'residual')
    opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) primal<opts.tol && dual<opts.tol; 
end

if strcmp(opts.stopRule,'iterations')
    opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) iter > opts.maxIters; 
end


% Default behavior:  Compute residual at this iteration and divide by
% maximum residual over all iterations.  Terminate when this ratio gets
% small
if strcmp(opts.stopRule,'ratioResidual')
    opts.stopNow = @(x,y,primal,dual,maxPrimal,maxDual) (primal/maxPrimal<opts.tol && dual/maxDual<opts.tol) || (primal<1e-10 && dual<1e-10); 
end


assert(isfield(opts,'stopNow'),['Invalid choice for stopping rule: ' opts.stopRule ]);





return

