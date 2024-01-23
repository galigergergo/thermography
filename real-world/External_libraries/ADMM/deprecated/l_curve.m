function [reg_corner,rho,eta,reg_param] = l_curve(A,b,npoints,nit)
%L_CURVE Plot the L-curve and find its "corner" in case of L1 penalty.

% Set defaults.
if nargin<4
    nit = 100;  % Number of iterations for the basis pursuit.
end
if nargin<3
    npoints = 50;  % Number of points on the L-curve for Tikh and dsvd.
    nit = 100;  % Number of iterations for the basis pursuit.
end
smin_ratio = 16*eps;  % Smallest regularization parameter.

% Initialization.
eta = zeros(npoints,1); rho = eta; 

normA=normest(A);
%reg_param(npoints) = normA*smin_ratio;
%ratio = (normA/reg_param(npoints))^(1/(npoints-1));
%for i=npoints-1:-1:1, reg_param(i) = ratio*reg_param(i+1); end
reg_param=flip(logspace(-2,round(log10(normA)),npoints));

%options.verbosity=0;
%options.iterations=30;
options  = IRcgls('defaults');
for i=1:npoints
    %x_reg=(A'*A + reg_param(i)* eye(size(A)))\(A'*b);
    %x_reg=(A'*A + reg_param(i)* eye(size(A)))\(A'*b);    
    %x_reg=spg_bpdn(A,b,reg_param(i),options);
    options = IRset(options,'IterBar', 'off', 'verbosity', 'off', 'Regparam',reg_param(i),'MaxIter',nit);
    [x_reg,info] = IRcgls(A,b,options);  
    i
    eta(i) = norm(x_reg,2);
    rho(i) = norm(A*x_reg-b,2);
end

% Locate the "corner" of the L-curve, if required.
reg_corner = l_corner(rho,eta,reg_param);
