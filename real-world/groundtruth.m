function groundtruth(groundtruth_file, reconstruction_dir, fnames, deg, Nx, Ny, Nz, thrs)
    files=dir(reconstruction_dir);
    vol=zeros(Ny,length(files)-2,Nz);
    [Nm,Nn]=size(imread([reconstruction_dir,fnames,num2str(2),'.png'])); 
    padding_z=round((Nm-Nz)/2);
    padding_y=round((Nn-Ny)/2);        
    %ROI_z=padding_z+(1:Nz);        
    ROI_z=1:Nz;
    ROI_y=padding_y+(1:Ny);   
    top=1;
    for i=1:length(files)-4
        img=imread([reconstruction_dir,fnames,num2str(i+1),'.png']); 
        binimg=img(ROI_z, ROI_y)>thrs;
        vol(:,end-i,:)=binimg';
    end
    x=linspace(0,40,Nx);
    y=linspace(0,40,Ny);
    z=linspace(0,8,Nz);
    load([groundtruth_file,'Data_down']); %loading the downscaled groundtruth volume
    Data_down=Data_down(:,end-1:-1:1,end:-1:1);
    [M,P,Q] = size(Data_down);
    xx=linspace(0,40,P);
    yy=linspace(0,40,M);
    zz=linspace(0, 8,Q);

    %% Counterclockwise rotation of the specimen
    [X,Y,Z]=meshgrid(xx-40/2,yy-40/2,zz-8/2);
    G=[cosd(-deg), -sind(-deg) 0; sind(-deg), cosd(-deg), 0; 0, 0, 1];
    rotgrid=G*[X(:), Y(:), Z(:)]';
    rotgrid_x=reshape(rotgrid(1,:),M,P,Q);
    rotgrid_y=reshape(rotgrid(2,:),M,P,Q);
    rotgrid_z=reshape(rotgrid(3,:),M,P,Q);
    Data_down_rot = interp3(X,Y,Z,Data_down,rotgrid_x,rotgrid_y,rotgrid_z,'linear',Data_down(1,1,1));
    
    %% Reflection along the Y and Z axis
    Data_down_rot = Data_down_rot(:,end:-1:1,end:-1:1);
    vol = vol(:,end:-1:1,end:-1:1);    
    
    figure(2)
    p1=patch(isosurface(xx,yy,zz,Data_down_rot));
    p1.FaceColor = 'blue';
    p1.EdgeColor = 'none';
    p1.FaceAlpha = 0.15;
    hold on;
    p2=patch(isosurface(x,y,z,vol));
    p2.FaceColor = 'red';
    p2.EdgeColor = 'none';
    camlight 
    lighting gouraud
    %plotting the boundaries
    circx=20*([cos(pi*(linspace(0,1,256))),-cos(pi*(linspace(0,1,256)))])+20;
    circy=20*([sin(pi*(linspace(0,1,256))),-sin(pi*(linspace(0,1,256)))])+20;
    circz=zeros(1,length(circx));
    plot3(circx,circy,circz,'k','LineWidth',2);
    plot3(circx,circy,circz+4,'k','LineWidth',1);
    plot3(circx,circy,circz+8,'k','LineWidth',2);
    hold off;
    axis equal
    grid on;
    xlabel('x [mm]');
    ylabel('y [mm]');
    zlabel('z [mm]');
    view(gca,[-30,30]);
    set(gca,'ztick',[0 4 8]);
    set(gca,'zticklabel',[8 4 0]);
    %set(gca,'ytick',[1 10 20 30 39]); set(gca,'yticklabel',[40 30 20 10 0])
    set(gca,'ytick',0:10:40);
    set(gca,'xtick',0:10:40);
    set(gca,'FontSize',18);

    figure(3)
    p1=patch(isosurface(xx,yy,zz,Data_down_rot));
    p1.FaceColor = 'blue';
    p1.EdgeColor = 'none';
    p1.FaceAlpha = 0.15;
    hold on;
    p2=patch(isosurface(x,y,z,vol));
    p2.FaceColor = 'red';
    p2.EdgeColor = 'none';
    camlight 
    lighting gouraud
    %plotting the boundaries
    circx=20*([cos(pi*(linspace(0,1,200))),-cos(pi*(linspace(0,1,200)))])+20;
    circy=20*([sin(pi*(linspace(0,1,200))),-sin(pi*(linspace(0,1,200)))])+20;
    circz=zeros(1,length(circx));
    plot3(circx,circy,circz,'k','LineWidth',2);
    plot3(circx,circy,circz+4,'k','LineWidth',1);
    plot3(circx,circy,circz+8,'k','LineWidth',2);
    plot3([20,20],[0 40],[0,0],'g--','LineWidth',2); %for indicating the scanning direction        
    hold off;
    axis equal;
    axis square;
    grid on;
    xlabel('x [mm]');
    ylabel('y [mm]');
    zlabel('z [mm]');
    view(gca,[0,90]);
    set(gca,'ztick',[0 4 8]);
    set(gca,'zticklabel',[8 4 0]);    
    set(gca,'ytick',0:10:40);
    set(gca,'xtick',0:10:40);
    set(gca,'FontSize',18);
    
    figure(4)
    subplot(2,1,1);
    p1=patch(isosurface(xx,yy,zz,Data_down_rot));
    p1.FaceColor = 'blue';
    p1.EdgeColor = 'none';
    p1.FaceAlpha = 0.1;
    hold on;
    p2=patch(isosurface(x,y,z,vol));
    p2.FaceColor = 'red';
    p2.EdgeColor = 'none';
    camlight 
    lighting gouraud
    %plotting the boundaries
    circx=20*([cos(pi*(linspace(0,1,200))),-cos(pi*(linspace(0,1,200)))])+20;
    circy=20*([sin(pi*(linspace(0,1,200))),-sin(pi*(linspace(0,1,200)))])+20;
    circz=zeros(1,length(circx));
    plot3(circx,circy,circz,'k','LineWidth',2);
    plot3(circx,circy,circz+4,'k','LineWidth',1);
    plot3(circx,circy,circz+8,'k','LineWidth',2);
    hold off;
    axis equal;
    grid on;
    xlabel('x [mm]');
    ylabel('y [mm]');
    zlabel('z [mm]');
    view(gca,[0,0]);
    axis tight;
    set(gca,'ztick',[0 4 8]);
    set(gca,'zticklabel',[8 4 0]);    
    set(gca,'ytick',[1 10 20 30 39]); set(gca,'yticklabel',[40 30 20 10 0])
    set(gca,'xtick',0:10:40);
    set(gca,'FontSize',18);

    subplot(2,1,2);
    p1=patch(isosurface(xx,yy,zz,Data_down_rot));
    p1.FaceColor = 'blue';
    p1.EdgeColor = 'none';
    p1.FaceAlpha = 0.1;
    hold on;
    p2=patch(isosurface(x,y,z,vol));
    p2.FaceColor = 'red';
    p2.EdgeColor = 'none';
    camlight 
    lighting gouraud
    %plotting the boundaries
    circx=20*([cos(pi*(linspace(0,1,200))),-cos(pi*(linspace(0,1,200)))])+20;
    circy=20*([sin(pi*(linspace(0,1,200))),-sin(pi*(linspace(0,1,200)))])+20;
    circz=zeros(1,length(circx));
    plot3(circx,circy,circz,'k','LineWidth',2);
    plot3(circx,circy,circz+4,'k','LineWidth',1);
    plot3(circx,circy,circz+8,'k','LineWidth',2);
    hold off;
    axis equal;
    grid on;
    xlabel('x [mm]');
    ylabel('y [mm]');
    zlabel('z [mm]');
    view(gca,[-90,0]);
    axis tight;
    set(gca,'ztick',[0 4 8]);
    set(gca,'zticklabel',[8 4 0]);    
    set(gca,'ytick',[1 10 20 30 39]); set(gca,'yticklabel',[0 10 20 30 40])
    set(gca,'xtick',0:10:40);
    axis([0 40 1 39 0 8]);
    set(gca,'FontSize',18);

end
