function frage3_fin_fil_filt=remove_line_effects(frage3_fin_fil)
%    Nt=5000;
%    frage3_fin_fil=frage3_fin_fil(:,:,1:Nt);
    [Nx,Ny,Nt]=size(frage3_fin_fil);
    frage3_fin_fil_filt=zeros(Nx,Ny,Nt);

    %Measurement properties
    dt = 1/25;          % time resolution
    rr = 40e-0;         % diameter of specimen (mm)
    dx=rr/238;          % spatial resolution

    t=(0:Nt-1)*dt;
    x=(0:Nx-1)*dx;
    ft=1/dt;
    fx=1/dx;
    Ft=(-Nt/2:Nt/2-1)*(ft/Nt);
    Fx=(-Nx/2:Nx/2-1)*(fx/Nx);
    winx=gausswin(Nx,12);
    wint=gausswin(Nt,300)';
    centert=floor(Nt/2)+1;
    centerx=floor(Nx/2)+1;    
    %Removing line effects in each cross-section of the material
    for j=1:1:Ny-1
        Ik=squeeze(frage3_fin_fil(:,j,:));
        fftIk_orig=fftshift(fft2(Ik));
        fftIk_filt=fftIk_orig;    
        fftIk_filt(:,centert)=winx.*abs(fftIk_orig(:,centert)).*exp(1i * angle(fftIk_orig(:,centert)));
        fftIk_filt(centerx,:)=wint.*abs(fftIk_orig(centerx,:)).*exp(1i * angle(fftIk_orig(centerx,:)));
        frage3_fin_fil_filt(:,j,:)=real(ifft2(ifftshift(fftIk_filt)));

%         %Displaying the results for each crossection    
%         subplot(2,2,1);
%         imagesc(log(abs(fftIk_orig)+1));
%         ylabel('spatial frequencies (cycles per mm)');
%         xlabel('temporal frequencies (cycles per sec)');
%         title('Original log Spectra');
% 
%         subplot(2,2,2);
%         imagesc(log(abs(fftIk_filt)+1));
%         ylabel('spatial frequencies (cycles per mm)');
%         xlabel('temporal frequencies (cycles per sec)');
%         title('Filtered log Spectra');
%  
%         subplot(2,2,3);
%         imagesc(Ik);
%         title('Original image');
%         subplot(2,2,4);
%         imagesc(squeeze(frage3_fin_fil_filt(:,j,:)));
%         title('Filtered image');
%         drawnow;    
    end
end