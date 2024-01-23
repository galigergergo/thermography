%% This is a wrapper function for computing the FSAFT operator and its adjoint for ultrasound image reconstruction.
%% This function can be passed to regularization techniques implemented in the IR tools package [1].
%
%
%% Input Parameters:
% v    : vectorized form of the estimated image on which the operator should be applied. 
% mod  : decides whether the operator or its adjoint should be applied. 
% Nt   : number of samples in time.
% Ny   : number of samples along the scanning direction.
% dt   : temporal sampling period.
% dy   : spatial sampling interval along the scanning direction.
% c0   : speed of sound in the specimen.
%
%
%
%% Output Parameters:
% y    : if mod='notransp', then y=fsaft_adjoint(v). Here, the forward problem is described by the adjoint.  
%        if mod='transp', then y=fsaft(v).           Note that the inverse problem is described by the SAFT operators.
%      
%% References
% [1] S. Gazzola, P. C. Hansen, J. G. Nagy, IR Tools: a MATLAB package 
%     of iterative regularization methods and large-scale test problems,
%     Numerical Algorithms, pp. 1-39 (2018).
%
%Implemented by Peter Kovacs, 09/2018. (kovika@inf.elte.hu)

function y = ultrasound_reconstruction(v,mod,Nt,Ny,dt,dy,c0)
    V=reshape(v,Nt,Ny);

    switch mod
        case {'notransp',1}
            Y=fsaft_adjoint(V,dt,dy,c0);
            y=reshape(Y,Nt*Ny,1);
        case {'transp',2} 
            Y=fsaft_operator(V,dt,dy,c0); 
            y=reshape(Y,Nt*Ny,1);
        otherwise
            error('Invalid mod parameter. Only transp or notransp are allowed!');
    end
end

