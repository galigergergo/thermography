%% This is a wrapper function for computing the corresponding operator and its adjoint for thermographic image reconstruction.
%% This function can be passed to regularization techniques implemented in the IR tools package [1].
%
%
%% Input Parameters:
% v      : estimation of the reconstructed image on which the operator should be applied. 
% mod    : decides whether the operator or its adjoint should be applied: 
% Nt     : number of samples in time.
% Ny     : number of samples along the scanning direction.
% dt     : temporal sampling period.
% dy     : spatial sampling interval along the scanning direction.
% c0     : speed of sound in the probe.
% alfa   : thermal diffusivity.
% method : choose the reconstruction method: 'fsaft' or 'tsaft'
% k      : in case method='tsaft' 2k+1 terms will be summed up along the hyperbolae.
%
%
%
%% Output Parameters:
% y    : if mod='notransp', then y=IK*FSAFT'*v  
%        if mod='transp', then y=FSAFT*K'*v
%      
%% References
% [1] S. Gazzola, P. C. Hansen, J. G. Nagy, IR Tools: a MATLAB package 
%     of iterative regularization methods and large-scale test problems,
%     Numerical Algorithms, pp. 1-39 (2018).
%
%Implemented by Peter Kovacs, 09/2018. (kovika@inf.elte.hu)

function y = thermographic_reconstruction(v,mod,Nt,Ny,dt,dy,c0,alfa,method,k)
    persistent K; 
    persistent TSAFT;

    %Computing the Kernel matrix 'K'
    if isempty(K)
        K = kernel_matrix(zeros(Nt,Ny),dt,c0,alfa);
    end
    
    switch method
        case 'fsaft'
            V=reshape(v,Nt,Ny);
            switch mod
                case {'notransp',1}
                    Y=K*fsaft_adjoint(V,dt,dy,c0);
                    y=reshape(Y,Nt*Ny,1);
                case {'transp',2} 
                    Y=fsaft_operator(K'*V,dt,dy,c0); 
                    y=reshape(Y,Nt*Ny,1);
                otherwise
                    error('Invalid mod parameter. Only transp or notransp are allowed!');
            end

        case 'tsaft'
            if isempty(TSAFT)
                [~,TSAFT] = tsaft_operator(zeros(Nt,Ny),k,dy,dt,c0);
            end
            switch mod
                case {'notransp',1}
                    y=TSAFT'*v;
                    Y=K*reshape(y,Nt,Ny);
                    y=reshape(Y,Nt*Ny,1);
                case {'transp',2} 
                    Y=K'*reshape(v,Nt,Ny);
                    y=TSAFT*reshape(Y,Nt*Ny,1);
                otherwise
                    error('Invalid mod parameter. Only transp or notransp are allowed!');
            end
        otherwise
            error('Invalid method parameter. Only fsaft or tsaft are allowed!');
    end
end

