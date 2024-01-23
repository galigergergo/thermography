%% This function implements the frequency-domain SAFT algorithm.
%  We modified the original codes [1] in order to adopt the algorithm 
%  to the thermographic image reconstruction problem.
%
%
%% Input Parameters:
% U    : surface ultrasound image on which the FSAFT operator should be applied. 
% dt   : temporal sampling period.
% dy   : spatial sampling interval along the scanning direction.
% c0   : speed of sound in the specimen.
%
%% Output Parameters:
% V    : reconstructed image using the FSAFT algorithm, i.e. V=FSAFT(U)=FSAFT*U.
%        The matrix FSAFT is not provided here, since it would be incredibly large due to the 2D FFT operator.
%        Instead, we are applying the FSAFT operator implicitly, i.e. apply
%        fast Fourier transform using the built-in MatLab routine 'fft2'.      
%      
%
%% References
% [1] Garcia D., Tarnec L. L., Muth, S., Montagnon, E., Poreé, J., Cloutier, G., 
%     Stolt's f-k Migration for Plane Wave Ultrasound Imaging, IEEE Transactions 
%     on Ultrasonics, Ferroelectrics, and Frequency Control, vol. 60, no. 9, pp. 1853-1867 (2013). 
%
%Modified by Peter Kovacs, 08/2018. (kovika@inf.elte.hu)

function V = fsaft_operator(U,dt,dy,c0)

    % Exploding Reflector Model velocity 
    [nt,ny] = size(U);
    persistent INT;
    
    % FFT
    fftRF = fftshift(fft2(U,nt,ny))/nt/ny;
    
    % Nearest neighbour interpolation  
    if isempty(INT)
        fs=1/dt; %Sampling frequency of the ultrasound signals.
        f = (-nt/2:nt/2-1)*fs/nt;
        kx = (-ny/2:ny/2-1)/dy/ny;
        [kx,f] = meshgrid(kx,f);
        fkz = c0*sign(f).*sqrt(kx.^2+f.^2/c0^2);          
        [nearestind,colind]=nearest(f,fkz);        
        INT=sparse(colind,nearestind,ones(1,length(nearestind)),ny*nt,ny*nt); 
    end
    fftRF2=reshape(INT*reshape(fftRF,nt*ny,1),nt,ny);  
    
     % IFFT & Migrated U
     V = ifft2(ifftshift(fftRF2));
end

% --------------------------Auxiliary functions-------------------------- %

% Nearest neighbour interpolation along hyperbolas.
function [nearestind,colind]=nearest(f,fkz) 
    [nf,mf]=size(f);
    l=log2(nf);
    nearestind=zeros(1,nf*mf);
    colind=zeros(1,nf*mf);
    top=0;

    for k=1:mf
        for i=1:1:nf
            ind=0;
            if fkz(i,k)>=f(1,k) && f(end,k)>=fkz(i,k)
                for j=1:1:l
                    if fkz(i,k)>=f(ind+2^(l-j),k)
                        ind=ind+2^(l-j);
                    end
                end
                if abs(fkz(i,k)-f(ind,k))<abs(fkz(i,k)-f(ind+1,k))
                    nearestind(top+1)=(k-1)*nf+ind;
                    colind(top+1)=(k-1)*nf+i;
                else
                    nearestind(top+1)=(k-1)*nf+ind+1;                    
                    colind(top+1)=(k-1)*nf+i;                    
                end
                top=top+1;
            end
        end
        
    end
    nearestind(top+1:end)=[];
    colind(top+1:end)=[];    
end
