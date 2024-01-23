%% This is a wrapper function for computing the operator 'IK' (bold K in the paper) and its adjoint for virtual wave reconstruction.
%% The matrix IK=kron(I,K) where I=eye(Ny,Ny) and 'K' is defined in [2].
%% This function can be passed to regularization techniques implemented in the IR tools package [1].
%
%
%% Input Parameters:
% u      : vectorized form of the estimated image on which the operator should be applied. 
% mod    : decides whether the operator or its adjoint should be applied. 
% mu     : penalty parameter 
% K      : the K matrix in the virtual wave reconstruction process [1].
% D,V    : these are the D,V matrices from the singular value decomposition of K=UDV'.
% Nt     : number of samples in time.
% Ny     : number of samples along the scanning direction.
%
%
%
%
%% Output Parameters:
% b          : if mod='notransp', then b=IK*u  
%              if mod='transp', then b=IK'*u
%              if 0<=mu, then d=(mu*I + IK^T*IK)^(-1) * u 
%      
%% References
% [1] S. Gazzola, P. C. Hansen, J. G. Nagy, IR Tools: a MATLAB package 
%     of iterative regularization methods and large-scale test problems,
%     Numerical Algorithms, pp. 1-39 (2018).
%
% [2] P. Burgholzer, M. Thor, J. Gruber, G. Mayr, Three-dimensional 
%     thermographic imaging using a virtual wave concept,
%     Journal of Applied Physics, vol. 121, issue 10, (2017).
%
%     Implemented by Peter Kovacs, 09/2018. (kovika@inf.elte.hu)

function [b] = virtualwave_reconstruction(u,mod,mu,K,D,V,Nt,Ny) 
    if nargin<9
        TSAFT=[];
    end
    d2=diag(D).^2;
    switch mod
        case {'notransp', 1}
            b=reshape(K*reshape(u,Nt,Ny),Nt*Ny,1);
        case {'transp', 2}
            b=reshape(K'*reshape(u,Nt,Ny),Nt*Ny,1);
        case 'invmuKtK'
            invmuKtK=V*(diag(1./(d2+mu)))*V';
            b=invmuKtK*u;
        otherwise
            error('Invalid mod parameter. Only invmuKtK, transp or notransp are allowed!');
    end
end

