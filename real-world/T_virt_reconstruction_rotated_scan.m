%% Estimating the virtual waves from the surface measurements by solving:
%
%               K*T_virt(:,y_j)=T(:,y_j)        (j=0,...,Ny-1), 
%

function T_virt_reconstruction_rotated_scan(inputdir,params)
load([inputdir,'steelRodsInEpoxy_rot_',num2str(params.theta),'.mat']);
%% Setting parameters of the specimen
Nit=20;                                 % Number of iterations for the ADMM algorithm.
%[Ny,Ns,Nt]=size(frage3_fin_fil_rot);    % Ny, Ns, Nt denote the number of discretization points in each direction.
Nt=params.Nt;
Ny=params.Ny;
c0=params.c0;                           % Dimensionless virtualwave speed.
dt = params.dt;                         % time resolution
alpha = params.alpha;                   % thermal diffusion
dx=params.dx;                           % spatial resolution
Nx=params.Nx;                           % number of discretization points along the depth dimension

%% Constructing the K matrix
display('Constructing the K matrix. This will take some time...');

T_noisy=squeeze(frage3_fin_fil_rot(:,100,1:params.Nt))'; % T is the measured surface temperature.
K = kernel_matrix(T_noisy,dt,dx,c0,alpha);

%% Constructing the Kabel matrix
[Kabel, normKabel, invabel] = Abel_trf(K);
K_ADMM = Kabel; 
At_Kabel=@(x) Kabel'*x;
[U_Kabel,s_Kabel,V_Kabel]=csvd(Kabel); 


surface=squeeze(frage3_fin_fil_rot(:,:,100));
for i=1:params.Ns
    display(sprintf('Cross-section %d',i));
    T_noisy=squeeze(frage3_fin_fil_rot(:,i,1:params.Nt))'; % T is the measured surface temperature.                  

    %% Estimating the virtual wave by using Abel transformation (Abel)
    reg_c=reg_param(T_noisy,U_Kabel,diag(s_Kabel));
    cADMM_Kabel = reg_c*1e-0;      % regularization parameter
    invmuAtA=@(x) virtualwave_reconstruction(x,'invmuKtK',cADMM_Kabel^2,[],diag(s_Kabel),V_Kabel,params.Nt,params.Ny);
    T_virt_abel = ADMM(T_noisy, invmuAtA, At_Kabel, cADMM_Kabel,'sp', Nit);

    T_virt_in_abelspace = T_virt_abel;
    T_virt_in_abelspace(T_virt_in_abelspace<0)=0;
    
    T_virt_abel = invabel*T_virt_abel/normKabel;

    T_virtual_abel = [];
        
    save(sprintf('%s/MAT/virtualwave_%d.mat',inputdir,i),'T_noisy','T_virtual_abel','T_virt_in_abelspace');   
    imwrite(mat2gray(T_virt_abel(1:Nx,:)),sprintf('%s/PNG/VIRT/virtualwave_%d.png',inputdir,i));   
    imwrite(mat2gray(T_noisy),sprintf('%s/PNG/SURF/surface_measurements_%d.png',inputdir,i)); 
end