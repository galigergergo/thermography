%F-SAFT Algorithm
%  [pxyzc] = ffsaft(prt,xi,yi,tt,c)
%
%  Returns the SAFT reconstructed initial (t=0) 3-d pressure data
%  reconstructed from time dependent 2-d pressure data
%
%  pxyzc ... p(x,y,z): the reconstructed inital 3-d pressure data
%  prt ..... p(x,y,z=0,t): 2-d time dependent pressure data
%  xi ...... x-vector of measurement points
%  yi ...... y-vector of measurement points
%  t ....... vector of measurement times
%  c ....... sound of speed in medium
%
% Author: Thomas Berer, Peter Burgholzer




function [pxyzc] = ffsaft1(prt,xi,yi,tt,c)

%Calculate A(kx,ky,z=0,w) with descrete cosine transform
%note: FFT and DCT gives k/(2*pi) and f, and not k and w!
display('... Calculating A(kx,ky,z=0,w) with dct ohne fftshift');
    tic;
   Akk0wc=zeros(size(prt));
    for ii=1:1:size(Akk0wc,3)
        Akk0wc(:,:,ii)=fft2(prt(:,:,ii));
    end
    for xx=1:size(Akk0wc,1)
        for yy=1:size(Akk0wc,2)
            Akk0wc(xx,yy,:)=dct(Akk0wc(xx,yy,:));
        end
    end
    toc;
    
    
%Get spatial and time sampling rate
Nx=length(xi);
Ny=length(yi);
Nt=length(tt);
fsx=(max(xi)-min(xi))/(Nx-1);
fsy=(max(xi)-min(xi))/(Ny-1);
tst=(max(tt)-min(tt))/(Nt-1);
% dtst=max(tt)-min(tt);

kx=[0:1/Nx:(1/2-1/Nx) -1/2:1/Nx:-1/Nx];
ky=[0:1/Ny:(1/2-1/Ny) -1/2:1/Ny:-1/Ny];
ww=0:1/2/Nt:1/2/Nt*(Nt-1);


display('... Calculating A(kx,ky,z=fixed,w) cosine');
tic


%Calculate the back propagator: exp(i*2*pi* z * sqrt((w(c)^2-kx^2-ky^2) )
%a facor of 2*pi is needed as we have k/(2*pi) and f vectors
expoc=zeros(size(Akk0wc));
for ikx=1:length(kx)
     for iky=1:length(ky)
             %Calculate part in square root
             expoc(ikx,iky,:)=ww.^2-kx(ikx)^2-ky(iky)^2;
             %eliminate non propagating modes
             %expoc=expoc.*heavy(expoc);
     end
end
%Calculate z-independent part of exp(%)
lpos=(expoc>=0);
expoc=2*pi*i*sqrt(expoc);
toc

display('Calculating p(x,y,z,0) ...');
%tic

tbp=0;
tazw=0;
tinv=0;

factor=exp(expoc).*lpos;
expo=ones(size(expoc));

%Calculate inital pressure data: p(x,y,z,t=0)
pxyzc=zeros(length(xi),length(yi),max(length(xi),length(yi)));
for z=1:1:size(pxyzc,3)
    %Calculate back propagator
    tic;
    expo=expo.*factor;
%     expo=real(expo);
    tbp=tbp+toc;
    
    %Calculate A(kx,ky,z,w)
    tic;
%     Akkzwc=real(expo).*Akk0wc;
    Akkzwc=expo.*Akk0wc;
    tazw=tazw+toc;
    
    %for time independent solution (for t=0) the idct reduces to
    %the sum over all frequencies
    tic;
    Akkzt0=sum(Akkzwc,3);
    
    %do inverse fft to get reconstucted data p(x,y,z,t=0)
    pxyzc(:,:,z)=real(ifft2(Akkzt0));
    tinv=tinv+toc;
end
%toc
display(['calc time backpropagator: ' num2str(tbp)]);
display(['calc time Akkzw: ' num2str(tazw)]);
display(['calc time inversion: ' num2str(tinv)]);


   
