%% This script runs the thermographic image reconstruction on the ROTATED real data.
%
%% Loading external libraries
addpath('./External_Libraries/Migration');
addpath('./External_Libraries/ADMM');
addpath('./External_Libraries/RegTool');
addpath('./External_Libraries/IRtools-master');

%% Parameters of the measurement data
params.Ny=238;
params.Ns=238;                                           % Ny, Ns, Nt denote the number of discretization points in each direction.
params.Nt=2000;
params.c0=1;                                             % Dimensionless virtualwave speed.
params.dt = 1/25;                                        % time resolution
rho = 1300;                                              % density
cp = 1700;                                               % specific heat capacity
k = 0.18;                                                % thermal conductivity
params.alpha = k/(rho*cp);                               % thermal diffusion
rr = 40e-3;                                              % diameter of specimen (m)
params.dx=rr/238;                                        % spatial resolution
params.dy=rr/238;                                        % spatial resolution
params.Nx=round(8e-3/params.dx);                         % number of discretization points along the depth dimension
params.theta=45;                                          % rotation angle (counterclockwise)
filter_on=true;

%% Generating the clear data
if filter_on
    rotated_measurement=['./RotatedData_filt_',num2str(params.Nt),'/Deg_',num2str(params.theta),'/'];
    rotated_virtualwaves=['./RotatedData_filt_',num2str(params.Nt),'/Deg_',num2str(params.theta),'/PNG/'];
else
    rotated_measurement=['./RotatedData/Deg_',num2str(params.Nt),num2str(params.theta),'/'];
    rotated_virtualwaves=['./RotatedData/Deg_',num2str(params.Nt),num2str(params.theta),'/PNG/'];
end
mkdir(rotated_measurement);
mkdir([rotated_measurement,'MAT']);
mkdir([rotated_measurement,'PNG']);
mkdir([rotated_measurement,'PNG/SURF']);
mkdir([rotated_measurement,'PNG/VIRT']);
mkdir([rotated_measurement,'PNG/FKMIG']);
groundtruth_file=[rotated_measurement,''];

display('Rotating and filtering the real measurement data.');
tic;
rotate_data('./Data/',rotated_measurement,params,filter_on);
exectime_filt=toc;
display('-----------------------------------');

display('Reconstructing the virtual waves.');
tic;
T_virt_reconstruction_rotated_scan(rotated_measurement,params);
exectime_vw=toc;
display('-----------------------------------');

display(sprintf('Execution time of the preprocessing step: %.2f sec',exectime_filt));
display(sprintf('Execution time of the virtual wave extraction: %.2f sec',exectime_vw));





