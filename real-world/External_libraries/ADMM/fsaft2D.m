function Tsaft = fsaft2D(c,T3)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% FSAFT reconstruction

prt = T3';
xi=1:size(prt,1);
tt = 1:size(prt,2);

%Get spatial and time sampling rate
Nx=length(xi);
Nt=length(tt);
fsx=(max(xi)-min(xi))/(Nx-1);
tst=(max(tt)-min(tt))/(Nt-1);

kx=[0:1/Nx:(1/2-1/Nx) -1/2:1/Nx:-1/Nx];
ww=0:1/2/Nt:1/2/Nt*(Nt-1);
ww=ww/c;

tic
   Ak0wc=fft(prt);
   Ak0wc=dct(Ak0wc')';
display('... Calculating A(kx,y=fixed,w) cosine');
% Calculate the back propagator: exp(i*2*pi* y * sqrt((w(c)^2-kx^2) )
% a factor of 2*pi is needed as we have k/(2*pi) and f vectors
expoc=zeros(size(Ak0wc));
for ikx=1:length(kx)
             %Calculate part in square root
             expoc(ikx,:)=ww.^2-kx(ikx)^2;
end
%Calculate z-independent part of exp(%)
lpos=(expoc>=0);
expoc=2*pi*i*sqrt(expoc);

display('Calculating p(x,y,0) ...');

factor=exp(expoc).*lpos;
expo=ones(size(expoc));

%Calculate inital pressure data: p(x,y,t=0)
pxyc=zeros(size(prt));
for y=1:1:size(pxyc,2)
    %Calculate back propagator
    expo=expo.*factor;
    %Calculate A(kx,y,w)
    Akywc=expo.*Ak0wc;
    
    %for time independent solution (for t=0) the idct reduces to
    %the sum over all frequencies
    Akyt0=sum(Akywc,2);
    
    %do inverse fft to get reconstucted data p(x,y,z,t=0)
    pxyc(:,y)=real(ifft(Akyt0));
end
toc

Tsaft=max(pxyc,0);
end

