%% This function computes the kernel matrix 'K' for thermographic image reconstruction described in [1].
%
%
%% Input Parameters:
% T_virt : virtual waves on which the kernel should be applied.
% Nt     : number of samples in time.
% Ny     : number of samples along the scanning direction.
% dt     : temporal sampling period.
% dz     : spatial (depth) resolution.
% c0     : speed of sound in the specimen.
% alfa   : thermal diffusivity. 
%
%
%% Output Parameters:
% K    : Nt x Nt kernel matrix.  
% T    : the operator is applied on T_virt, i.e. T=K*T_virt.  
%      
%% References
% [1] P. Burgholzer, M. Thor, J. Gruber, G. Mayr, Three-dimensional 
%     thermographic imaging using a virtual wave concept,
%     Journal of Applied Physics, vol. 121, issue 10, (2017).
%
%Implemented by Peter Kovacs, 09/2018. (kovika@inf.elte.hu)

function [K, T] = kernel_matrix(T_virt,dt,dz,c0,alfa)
    [Nt,Ny]=size(T_virt);
    fourier = alfa * dt / dz^2; %Dimensionless Fourier number.
    tt=(0:Nt-1);
    K=(c0./sqrt(pi*fourier*tt')*ones(1,Nt)).*exp(-c0^2/4/fourier./tt'*tt.^2);    
    K(1,:)=0;
    
    if 2==nargout
        T=K*T_virt;
    end
end

