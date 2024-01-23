function plot_subproblem(T,i,T_virt,xd)
    subplot(2,2,1);
    imagesc(T);
    hold on;
    plot([i i],[0 size(T,1)],'b');
    hold off;
    xlabel('t','fontsize',12);
    ylabel('y','fontsize',12);
    set(gca,'fontsize',12);
    grid
    axis image
    axis xy
    colormap gray
    title('Surface temperature','FontSize',12);
    
    subplot(2,2,2);
    imagesc(T_virt);
    hold on;
    plot([i i],[0 size(T_virt,1)],'b');
    hold off;
    xlabel('t','fontsize',12);
    ylabel('y','fontsize',12);
    set(gca,'fontsize',12);
    grid
    axis image
    axis xy
    colormap gray
    title('Virtual wave image','FontSize',12);
    
    subplot(2,2,[3 4]);
    plot(T_virt(:,i),'b');
    hold on;
    plot(xd,'r--');
    hold off;
    xlabel('y','fontsize',12);
    ylabel('Tvirt','fontsize',12);
    set(gca,'fontsize',12);
    grid
    title('Virtual wave (regularized solution)','FontSize',12);
    
%    display('Press any key to continue...');
    pause(0.1);    
    drawnow;
end

