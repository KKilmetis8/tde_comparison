nstart = 308
nend = 309
if ischar(nstart)
  nstart=str2double(nstart);
end
if ischar(nend)
  nend=str2double(nend);
end
for j=nstart:1:nend %625
  clearvars -except j nstart nend
  out =  HealpixGenerateSampling(4, 'scoord');
  x=sin(out(:,1)).*cos(out(:,2));
  y=sin(out(:,1)).*sin(out(:,2));
  z=cos(out(:,1));
  X=[x y z]; % K: shape is 192 * 3, 1
  cross_dot = X*X';
  cross_dot(cross_dot<0)=0;
  cross_dot = cross_dot * 4 / 192;
  h=6.62607e-27;
  kb=1.38065e-16;
  
  Nray=5000;
  c=3e10;
  nu_min=kb*1e3/h;
  nu_max=kb*3e13/h;
  nnu=1000;
  a=7.5657e-15;
  nu=logspace(log10(nu_min), log10(nu_max), nnu);
  file_directory='home/Astro-Data/12R-TDE/goodcode';
  Tcool = dlmread(sprintf('%s/T.txt', file_directory));
  RhoCool = dlmread(sprintf('%s/rho.txt', file_directory));
  [Tinterp, Rhointerp] = meshgrid(Tcool, RhoCool);
  rossland = dlmread(sprintf('%s/ross.txt', file_directory));
  planck = dlmread(sprintf('%s/planck.txt', file_directory));
  % As far as I can tell, this does nothing.
  rossland = reshape(rossland, length(Tcool), length(RhoCool));
  planck = reshape(planck, length(Tcool), length(RhoCool));

  [Tcool2,RhoCool2,rossland2]=pad_interp(Tcool,RhoCool,rossland');
  [Tcool2,RhoCool2,planck2]=pad_interp(Tcool,RhoCool,planck');
         
 dirname='c:/sim_data';
 outdir=dirname;
  
  photo_plot=0;   
  j
  disp('Start reading');
  res=c(sprintf('%s/snap_%d.h5',dirname,j));

  X=[res.CMx res.CMy res.CMz];
  disp('Building tree');
  k=KDTreeSearcher(X);
  out =  HealpixGenerateSampling(4, 'scoord');  % k: again?
  
  rr=sqrt(res.CMx.^2+res.CMy.^2+res.CMz.^2);
  vr=(res.Vx.*res.CMx+res.Vy.*res.CMy+res.CMz.*res.Vz)./rr;
  clear rr;
  v=sqrt(res.Vx.^2+res.Vy.^2+res.Vz.^2);      
  Er=res.Density.*res.Erad;
  
  d_ray=zeros(Nray, length(out));
  r_ray=d_ray;
  T_ray=d_ray;
  Trad_ray=d_ray;
  Tnew_ray=d_ray;
  tau_ray=d_ray;
  tau_therm_ray=d_ray;
  v_ray=d_ray;
  c_ray=d_ray;
  d_photo=zeros(length(out),1);
  d_therm = d_photo;
  x_herm = d_photo;
  y_therm = d_photo;
  z_therm = d_photo;
  vr_therm = d_photo;
  T_photo=d_photo;
  r_photo=d_photo;
  r_therm=d_photo;
  photo_ind=d_photo;
  v_photo=d_photo;
  F_photo=zeros(length(out),nnu);
  F_photo_linear=zeros(length(out),nnu);
  F_photo_temp=zeros(length(out),nnu);
  F_photo_temp_linear=zeros(length(out),nnu);
  T_avg_photo=d_photo;
  L_avg = d_photo;
  L = d_photo;
  Lnew = d_photo;
  L_g = d_photo;
  L_r = d_photo;
  L_i = d_photo;
  L_uvw2 = d_photo;
  L_uvm2 = d_photo;
  L_uvw1 = d_photo;
  L_uvotu = d_photo;
  L_XRT = d_photo;
  L_XRT2 = d_photo;
     
  f_g = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Palomar_ZTF.g.dat'));
  f_g(:,1) = 3e18 ./ f_g(:,1);
  nu_g = trapz(f_g(:,1), f_g(:,1).*f_g(:,2)) / trapz(f_g(:,1), f_g(:,2));
  
  f_i = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Palomar_ZTF.i.dat'));
  f_i(:,1) = 3e18 ./ f_i(:,1);
  nu_i = trapz(f_i(:,1), f_i(:,1).*f_i(:,2)) / trapz(f_i(:,1), f_i(:,2));
  
  f_r = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Palomar_ZTF.r.dat'));
  f_r(:,1) = 3e18 ./ f_r(:,1);
  nu_r = trapz(f_r(:,1), f_r(:,1).*f_r(:,2)) / trapz(f_r(:,1), f_r(:,2));
  
  f_uvotu = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Swift_UVOT.U.dat'));
  f_uvotu(:,1) = 3e18 ./ f_uvotu(:,1);
  nu_uvotu = trapz(f_uvotu(:,1), f_uvotu(:,1).*f_uvotu(:,2)) / trapz(f_uvotu(:,1), f_uvotu(:,2));
  
  f_uvm2 = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Swift_UVOT.UVM2.dat'));
  f_uvm2(:,1) = 3e18 ./ f_uvm2(:,1);
  nu_uvm2 = trapz(f_uvm2(:,1), f_uvm2(:,1).*f_uvm2(:,2)) / trapz(f_uvm2(:,1), f_uvm2(:,2));
  
  f_uvw2 = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Swift_UVOT.UVW2.dat'));
  f_uvw2(:,1) = 3e18 ./ f_uvw2(:,1);
  nu_uvw2 = trapz(f_uvw2(:,1), f_uvw2(:,1).*f_uvw2(:,2)) / trapz(f_uvw2(:,1), f_uvw2(:,2));
  f_uvw1 = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\Swift_UVOT.UVW1.dat'));
  f_uvw1(:,1) = 3e18 ./ f_uvw1(:,1);
  nu_uvw1 = trapz(f_uvw1(:,1), f_uvw1(:,1).*f_uvw1(:,2)) / trapz(f_uvw1(:,1), f_uvw1(:,2));
  
  f_XRT = flipud(dlmread('C:\Users\Elads\Dropbox\TDE_Nick\XRT.dat'));
  f_XRT(:,1) = 3e18 ./ f_XRT(:,1);
  nu_XRT = trapz(f_XRT(:,1), f_XRT(:,1).*f_XRT(:,2)) / trapz(f_XRT(:,1), f_XRT(:,2));
  
  a=7.5657e-15;
  for i=1:length(out)
      if(mod(i,10)==0)
          disp(sprintf('Angle number %d',i));
      end
      
      mu_x = sin(out(i,1))*cos(out(i,2));
      mu_y = sin(out(i,1))*sin(out(i,2));
      mu_z = cos(out(i,1));
      
      if(mu_x < 0)
          rmax = res.Box(1) / mu_x;
      else
          rmax = res.Box(4) / mu_x;
      end
      if(mu_y < 0)
          rmax = min(rmax, res.Box(2) / mu_y);
      else
          rmax = min(rmax, res.Box(5) / mu_y);
      end
      if(mu_z < 0)
          rmax = min(rmax, res.Box(3) / mu_z);
      else
          rmax = min(rmax, res.Box(6) / mu_z);
      end
      
      r=logspace(-0.25,log10(rmax), Nray);
      alpha=(r(2)-r(1))./(0.5*(r(1)+r(2)));
      dr=alpha*r;
      
      x=r*mu_x;
      y=r*mu_y;
      z=r*mu_z;
      X2=[x; y; z]';
      idx=knnsearch(k,X2);
      d=res.Density(idx)*2e33/(7e10)^3;
      t=res.Temperature(idx); 
      sigma_rossland=exp(interp2(Tcool2,RhoCool2,rossland2,log(t),log(d),'linear',0));
      sigma_planck=exp(interp2(Tcool2,RhoCool2,planck2,log(t),log(d),'linear',0));

      los=-flipud(cumtrapz(flipud(r'),flipud(sigma_rossland)))*7e10;
      los_abs=-flipud(cumtrapz(flipud(r'),flipud(sigma_planck)))*7e10;
      los_effective=-flipud(cumtrapz(flipud(r'),sqrt(3*flipud(sigma_planck).*flipud(sigma_rossland))))*7e10;
      
      tau_tot = dr'.*7e10.*sigma_rossland;
      idxnew = knnsearch(k,[res.CMx(idx) res.CMy(idx) res.CMz(idx)],'K',20);
      idxnew = idxnew(:); % this is a .T WHY
      idxnew = unique(idxnew);
      dx = 0.5 * res.Volume(idx).^(1/3);
      Finterp = scatteredInterpolant(res.CMx(idxnew),res.CMy(idxnew),res.CMz(idxnew),Er(idxnew));
      gradx = (Finterp(res.CMx(idx)+dx,res.CMy(idx), res.CMz(idx)) - Finterp(res.CMx(idx)-dx,res.CMy(idx), res.CMz(idx)))./(2*dx);
      grady = (Finterp(res.CMx(idx),res.CMy(idx)+dx, res.CMz(idx)) - Finterp(res.CMx(idx),res.CMy(idx)-dx, res.CMz(idx)))./(2*dx);
      gradz = (Finterp(res.CMx(idx),res.CMy(idx), res.CMz(idx)+dx) - Finterp(res.CMx(idx),res.CMy(idx), res.CMz(idx)-dx))./(2*dx);
      grad = sqrt(gradx.^2+grady.^2+gradz.^2);
      rhat = [sin(out(i,1))*cos(out(i,2)) sin(out(i,1))*sin(out(i,2)) cos(out(i,1))];
      gradr = (gradx.*rhat(1) + grady .* rhat(2) + gradz.*rhat(3));

      R_lambda = max(1e-10, grad ./ (7e10 * sigma_rossland .* Er(idx)));
      fld_factor = 3 * (1.0 ./ tanh(R_lambda) - 1.0 ./ R_lambda) ./ R_lambda;
      smoothed_flux = -movingmean(r'.^2.*fld_factor.*(gradr)./sigma_rossland,7);
      v_grad = gradx.*res.Vx(idx) + grady.*res.Vy(idx) + gradz.*res.Vz(idx);
      smoothed_edot=r'.^2.*fld_factor.*v_grad;
      smoothed_edot =flipud(cumtrapz(flipud(r'),flipud(smoothed_edot)));
      vvr = vr(idx);
      EEr = Er(idx);
      
      
      good_candidates = smoothed_flux > 0 & (los < 0.6666666666);
      idd = find(good_candidates==1);
      b = idd(1);         
      [~,b2]=min(abs(los_effective - 5));
      
      idphoto = idx(b);
      Lphoto2 = 4 * pi * c * smoothed_flux(b) *2e33 / (1603^2);
      if Lphoto2 <0
          Lphoto2 = 1e100;
      end
      Lphoto = min(Lphoto2,4 * pi * c * EEr(b) * r(b)^2*2e33*7e10 / (1603^2));
      
      good_candidates = (los < 0.6666666666);
      idd = find(good_candidates==1);
      b = idd(1);
      
      r_photo(i)=r(b);
      d_photo(i)=d(b);
      v_photo(i)=v(idphoto);
      photo_ind(i)=idx(b);
      
      Tr = (Er(idx)*2e33/(a*7e10*1603^2)).^0.25;   
      d_ray(:,i)=d;
      r_ray(:,i)=r;
      Tnew_ray(:,i)=t;
      Trad_ray(:,i)=Tr;
      tau_ray(:,i)=los;
      tau_therm_ray(:,i)=los_effective;

      for kk=b2:length(r)
           F_photo_temp(i,:)= F_photo_temp(i,:) + sigma_planck(kk).*exp(-min(30,los_effective(kk))).*nu.^3 ./ (c * c * (exp(h * nu /(kb * t(kk))) - 1));
      end
      F_photo_temp(i,:) = F_photo_temp(i,:) * Lphoto / trapz(nu, F_photo_temp(i,:));
      L(i)=trapz(nu, F_photo_temp(i,:));
      LL(i)=Lphoto2;
      [~,b2]=min(abs(los_effective - 1));
      r_therm(i) = r(b2);
      T_photo(i)=t(b2);
      
      d_therm(i) = d(b2);
      vr_therm(i) = vvr(b2);
  end
  clear vr;
 
% ------------------------- FITTING -----------------------------------        
  T_fit=0;
  L_fit=0;
  ft = fittype( 'a*x.^3./(exp(b.*x)-1)', 'independent', 'x', 'dependent', 'y' );
  opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
  opts.Display = 'Off';
  opts.StartPoint = [0.421761282626275 0.915735525189067];
  x=[nu_g nu_r nu_i nu_uvw2 nu_uvw1 nu_uvm2]/1e15;
  for i=1:length(out)
      F_photo(i,:) = cross_dot(i,:) * F_photo_temp(:,:);
      #% WHAT WE WANT
      
      T_avg_photo(i) = dot(T_photo.*L,cross_dot(i,:))./dot(L,cross_dot(i,:));
      L_avg(i) =  dot(L, cross_dot(i,:));
      L_g(i) = trapz(f_g(:,1), interp1(nu, F_photo(i,:), f_g(:,1)).*f_g(:,2)) / trapz(f_g(:,1), f_g(:,2));
      L_r(i) = trapz(f_r(:,1), interp1(nu, F_photo(i,:), f_r(:,1)).*f_r(:,2)) / trapz(f_r(:,1), f_r(:,2));
      L_i(i) = trapz(f_i(:,1), interp1(nu, F_photo(i,:), f_i(:,1)).*f_i(:,2)) / trapz(f_i(:,1), f_i(:,2));
      L_uvw2(i) = trapz(f_uvw2(:,1), interp1(nu, F_photo(i,:), f_uvw2(:,1)).*f_uvw2(:,2)) / trapz(f_uvw2(:,1), f_uvw2(:,2));
      L_uvw1(i) = trapz(f_uvw1(:,1), interp1(nu, F_photo(i,:), f_uvw1(:,1)).*f_uvw1(:,2)) / trapz(f_uvw1(:,1), f_uvw1(:,2));
      L_uvm2(i) = trapz(f_uvm2(:,1), interp1(nu, F_photo(i,:), f_uvm2(:,1)).*f_uvm2(:,2)) / trapz(f_uvm2(:,1), f_uvm2(:,2));
      L_uvotu(i) = trapz(f_uvotu(:,1), interp1(nu, F_photo(i,:), f_uvotu(:,1)).*f_uvotu(:,2)) / trapz(f_uvotu(:,1), f_uvotu(:,2));
      L_XRT(i) = -trapz(f_XRT(:,1), interp1(nu, F_photo(i,:), f_XRT(:,1)).*f_XRT(:,2)) / trapz(f_XRT(:,1), f_XRT(:,2));
      ratio = zeros(length(f_XRT(:,2)),1);
      ratio(f_XRT(:,2) > max(f_XRT(:,2)) * 0.2) = 1;
      L_XRT2(i) = -trapz(f_XRT(:,1), interp1(nu, F_photo(i,:), f_XRT(:,1)).*ratio);
      yy = [L_g(i) L_r(i) L_i(i) L_uvw2(i) L_uvw1(i) L_uvm2(i)];
      scale = yy(end);
      [xData, yData] = prepareCurveData( x, yy / scale );
      [fitresult, gof] = fit( xData, yData, ft, opts );
      if(fitresult.a < 0)
          format long;
          display('Bad fit');
          [L_g(i) L_r(i) L_i(i) L_uvw2(i) L_uvw1(i) L_uvm2(i)]/scale
          x
          scale
          fitresult.a
          fitresult.b
          error('stop');
      end
      T_fit(i) = h*1e15/(kb*fitresult.b);
      L_fit(i) = (a*T_fit(i)^4*3e10 / 4)*(3e10)^2*fitresult.a*scale/2/h/1e45;
  end
      Time=res.Time;
      save(sprintf('%s/results/data_%d.mat',outdir,j),'Trad_ray','Tnew_ray','r_ray','tau_ray','d_ray','T_ray','tau_therm_ray','r_therm','T_fit','L_fit','L_g','L_r','L_i','L_uvw2','L_XRT','L_XRT2','L_uvotu','L_uvw1','L_uvm2','Time','L','r_photo','T_photo','d_photo','v_photo','F_photo','F_photo_temp','T_avg_photo','L_avg','nu','d_therm','vr_therm','-v7.3');
  end
