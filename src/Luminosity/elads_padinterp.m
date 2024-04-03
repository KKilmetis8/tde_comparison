function [x,y,V]=pad_interp(x,y,V)
    [V,x]=linearpad(V,x);
    [V,x]=linearpad(fliplr(V),flip(x));
    V=V.';
    [V,y]=linearpad(V,y);
    [V,y]=linearpad(fliplr(V),flip(y));
    V=V.';
end

function [D,z]=linearpad(D0,z0)
    factor=100;
    
    dz=z0(end)-z0(end-1);
    dD=D0(:,end)-D0(:,end-1);
    
    z=z0;
    z(end+1)=z(end)+factor*dz;
    
    D=D0;
    D(:,end+1)=D(:,end) + factor*dD;
end
