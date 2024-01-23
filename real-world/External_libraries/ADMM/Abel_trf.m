function [Kabel, normKabel, invabel] = Abel_trf(K)
    normK=norm(K);
    K = K/normK;
    Nt=size(K,2);
    c1=1:Nt+1;
    for tr=1:Nt+1
        c2=real(sqrt((tr+1)^2-c1.^2));
        hlp1=atan2(c1,c2);
        abel(tr,:)=diff(hlp1);
    end
    abel = abel(1:Nt,:);
    invabel = inv(abel);
    Kabel = pi/2*normK*K * invabel;
    normKabel = norm(Kabel);
    Kabel = Kabel/normKabel;
end

