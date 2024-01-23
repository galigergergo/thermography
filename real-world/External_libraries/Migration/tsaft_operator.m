%% This function implements the time-domain SAFT algorithm according to [1].
%  We adopted the original algorithm to the thermographic image reconstruction problem.
%
%
%% Input Parameters:
% U     : measured ultrasound image on which the TSAFT operator should be applied. 
% 2k+1  : is the number of terms to sum up along the diffraction hyperbolas.
% dy    : spatial sampling interval along the scanning direction.
% dt    : temporal sampling period
% c0    : speed of sound in the specimen.
%
%% Output Parameters:
% V     : reconstructed image using the TSAFT algorithm, i.e. V=TSAFT*U.
% TSAFT : (sparse) matrix form of the TSAFT operator.
%
%% References
% [1] Lingvall F., Olofsson T., Stepinski T., Synthetic aperture imaging using 
%     sources with finite aperture: Deconvolution of the spatial impulse response,
%     The Journal of the Acoustical Society of America, vol. 114, pp. 225-234 (2003).
%
%Implemented by Peter Kovacs, 08/2018. (kovika@inf.elte.hu)

function [V,TSAFT] = tsaft_operator(U,k,dt,dy,c0)
    [Nt,Ny]=size(U);
    dx=dt*c0;
    sval=zeros(1,Nt*(k+1));
    scolind=zeros(1,Nt*(k+1));
    srowind=zeros(1,Nt*(k+1));
    lens=zeros(1,k+1);
    top=0;
    for i=0:k  
        [s,blockrowind,blockcolind] = S_d(k-i,0,k,Nt,dy,dx,dt,c0);
        sval(top+1:top+length(s))=s;
        scolind(top+1:top+length(s))=blockcolind;
        srowind(top+1:top+length(s))=blockrowind+i*Nt;                
        top=top+length(s);
        lens(i+1)=length(s);     
    end
    sval(top+1:end)=[];
    scolind(top+1:end)=[];
    srowind(top+1:end)=[];       

    sval_blockflip=zeros(1,top-lens(end));
    scolind_blockflip=zeros(1,top-lens(end));
    srowind_blockflip=zeros(1,top-lens(end));
    top=0;
    for i=k:-1:1
        ind=sum(lens(1:i-1));
        sval_blockflip(top+1:top+lens(i))=sval(ind+1:ind+lens(i));
        scolind_blockflip(top+1:top+lens(i))=scolind(ind+1:ind+lens(i));
        srowind_blockflip(top+1:top+lens(i))=srowind(ind+1:ind+lens(i))+2*(k+1-i)*Nt;                
        top=top+lens(i);     
    end
    
    sval=[sval sval_blockflip];
    scolind=[scolind scolind_blockflip];
    srowind=[srowind srowind_blockflip];
    bandwidth=length(sval);
    
    S_rows=repmat(srowind,1,Ny);
    S_cols=repmat(scolind,1,Ny);
    S_vals=repmat(sval,1,Ny);

    for i=1:Ny 
        S_cols((i-1)*bandwidth+1:i*bandwidth)=S_cols((i-1)*bandwidth+1:i*bandwidth)+(i-1)*Nt;
        S_rows((i-1)*bandwidth+1:i*bandwidth)=S_rows((i-1)*bandwidth+1:i*bandwidth)+(i-1)*Nt;
        S_vals((i-1)*bandwidth+1:i*bandwidth)=S_vals((i-1)*bandwidth+1:i*bandwidth);
    end
    TSAFT=sparse(S_rows,S_cols,S_vals, Nt*(2*k+Ny),Ny*Nt);
    u=reshape(U,Nt*Ny,1);
    v=TSAFT*u;
    V=reshape(v(Nt*k+1:end-Nt*k),Nt,Ny);
   
end

function [s,rowind,colind] = S_d(n_hat,n,K,Nt,dy,dx,dt,c0)
    s=zeros(1,Nt);    
    rowind=zeros(1,Nt);
    colind=zeros(1,Nt);
    top=0;
    for i=0:1:Nt-1
        x_m=i*dx;
        r=sqrt((((n_hat-n)*dy).^2+ (x_m).^2));
        j=round(r/c0/dt);        
        if j<=Nt-1
            s(top+1)=alpha(n_hat-n,K)/max([r,1]);
            rowind(top+1)=i+1;
            colind(top+1)=j+1;            
            top=top+1;
        end
    end
    s(top+1:end)=[];
    rowind(top+1:end)=[];
    colind(top+1:end)=[];
end

function a = alpha(d,K)
    %a=ones(size(d)); %rectangular window
    a=0.5*(1-cos(2*pi*(d+1+K)/(2*K+1))); %Hann window
end
