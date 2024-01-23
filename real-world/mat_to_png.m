for i=1:238
    disp(i);
    inputdir = './RotatedData_filt_2000/Deg_45/FISTANet';
    load(sprintf('%s/MAT/virtualwave_%d.mat',inputdir,i));
    padding = (256 - size(fistanet_virt_space, 2)) / 2;
    padded = zeros(64, 256);
    padded(:, padding:size(fistanet_virt_space, 2)+padding-1) = fistanet_virt_space(1:64,:);
    imwrite(mat2gray(padded),sprintf('%s/PNG/virtualwave_%d.png',inputdir,i));
end