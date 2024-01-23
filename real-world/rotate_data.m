function rotate_data(inputdir,outputdir, params, filter_on)
    load(sprintf('%s',inputdir,'steelRodsInEpoxy.mat'));
    %% Counterclockwise rotation of the specimen
    if 0~=params.theta
        [grid_y,grid_s,grid_t] = meshgrid(-params.Ny/2:params.Ny/2-1,-params.Ns/2:params.Ns/2-1,1:params.Nt);
        G=[cosd(params.theta), -sind(params.theta) 0; sind(params.theta), cosd(params.theta), 0; 0, 0, 1];
        rotgrid=G*[grid_y(:), grid_s(:), grid_t(:)]';
        rotgrid_y=reshape(rotgrid(1,:),params.Ny,params.Ns,params.Nt);
        rotgrid_s=reshape(rotgrid(2,:),params.Ny,params.Ns,params.Nt);
        rotgrid_t=reshape(rotgrid(3,:),params.Ny,params.Ns,params.Nt);
        frage3_fin_fil_rot = interp3(grid_y,grid_s,grid_t,frage3_fin_fil(:,:,1:params.Nt),rotgrid_y,rotgrid_s,rotgrid_t,'linear',frage3_fin_fil(1,1,1));
    else
        frage3_fin_fil_rot = frage3_fin_fil;
    end
    if filter_on
        frage3_fin_fil_rot = remove_line_effects(frage3_fin_fil_rot);
    end
    
    save([outputdir,'steelRodsInEpoxy_rot_',num2str(params.theta)],'frage3_fin_fil_rot','-v7.3');
end