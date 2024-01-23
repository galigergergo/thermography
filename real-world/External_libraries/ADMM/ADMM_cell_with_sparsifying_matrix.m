%% Sparse approximation by using sparsifying matrix
% Optimization problem formulation: 
%
%           arg min_x = 0.5*|| y - A x ||_2^2 + lambda ||Pt x||_1,
%
%where Pt denotes the sparsifying transform, e.g. curvelet trf.

%% Input Parameters:
% y        : measurement vector. 
% invmuAtA : function handle to inv(A^T A + lambda I).
% At       : function handle to the transpose of A.
% lambda   : regularization parameter.
% P        : sparsifying matrix.
% Pt       : transpose of P.
% mod      : defines the thresholding operator:
%            soft ('s'), positive ('p'), soft positive ('sp'), hard ('h') 
%
%
%% Output Parameters:
% MADMM    : solution to the sparse approximation problem.

function  MADMM = ADMM_cell_with_sparsifying_matrix(y,invmuAtA, At, lambda, P, Pt, mod, Nit)

zero_coeff_struct=zeros(size(y));
MADMM = At(zero_coeff_struct);
NADMM = Pt(zero_coeff_struct);
RADMM = Pt(zero_coeff_struct);

if nargin<7
    thrsop=@(x) thresh_markus(x, 's', 1);
else
    thrsop=@(x) thresh_markus(x, mod, 1);
end

if nargin<8
    Nit=10;
end
   
    for kADMM = 1:Nit 
        kADMM
        NdiffRADMM=binop2cell(NADMM,RADMM,@minus);  %NADMM-RADMM
        lambdaAtA=At(y) + lambda^2 * P(NdiffRADMM); %Aty + lambda^2*(NADMM-RADMM)
        MADMM  = invmuAtA(lambdaAtA);
        Pt_MADMM = Pt(MADMM);
        
        MRADMM = binop2cell(Pt_MADMM,RADMM,@plus); % Computing Pt_MADMM + RADMM
        NADMM  = unop2cell(MRADMM,thrsop);         % Computing thresh_markus(MRADMM, 'sp', 1);
        
        RADMM  = compute_lambdaAtA(RADMM,Pt_MADMM,NADMM,1); %Computing RADMM + Pt_MADMM - NADMM
    end 
end

% Applies a binary operator on cell1 and cell2.
function result=binop2cell(cell1,cell2,op)
    result=cell(size(cell1));
    for s = 1:length(cell1)
        result{s}=cellfun(op,cell1{s},cell2{s},'UniformOutput',false);
    end
end

% Applies an unary operator on cell1.
function result=unop2cell(cell1,op)
    result=cell(size(cell1));
    for s = 1:length(cell1)
        result{s}=cellfun(op,cell1{s},'UniformOutput',false);
    end
end

% Executes an operator between the corresponding elements of cell1 and cell2.
function result=compute_lambdaAtA(Aty,NADMM,RADMM,lambda)
    result=cell(size(Aty));
    for s = 1:length(Aty)
      for w = 1:length(Aty{s})
        result{s}{w} = Aty{s}{w} + lambda^2*(NADMM{s}{w}-RADMM{s}{w});
      end
    end
end