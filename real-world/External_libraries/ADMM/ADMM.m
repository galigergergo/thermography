function  MADMM = ADMM(y, invmuAtA, At, lambda, mod, Nit)
if nargin <5
    mod='s';
end

if nargin <6
    Nit = 10;
end

Ny=size(y,2);

MADMM = zeros(size(y,1),Ny);
NADMM = zeros(size(y,1),Ny);
RADMM = zeros(size(y,1),Ny);

    for kADMM = 1:Nit 
        MADMM =  invmuAtA(At(y) + lambda^2*(NADMM-RADMM));
        NADMM = thresh_markus(MADMM + RADMM, mod, 1);
        RADMM =  RADMM + MADMM - NADMM;
    end 
end

