function reg_c = reg_Param(T,matU,matD)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
%%
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% ++++++++++++++Estimation of Reg-Parameter for it. Thik. and DR ++++++++++
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

%%
[Nt,Ny]=size(T);
if size(T,1)~=1 && size(T,2)~=1 %if T is a matrix
    U=@(x) reshape(matU*reshape(x,Nt,Ny),Nt*Ny,1);
    Ut=@(x) reshape(matU'*reshape(x,Nt,Ny),Nt*Ny,1);    
    s=repmat(diag(matD),Ny,1);
    b = reshape(T,Nt*Ny,1);    
else
    U=@(x) matU*x;
    Ut=@(x) matU'*x;
    s=diag(matD);
    b = T(:,1);
end

%%
figure (15)
[reg_c,rho,eta,reg_param] = l_curve(U,Ut,s,b,'Tikh');

end

