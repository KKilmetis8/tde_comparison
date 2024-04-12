addpath('npy-matlab-master/npy-matlab')
savepath


path = '/home/konstantinos/Astro-Data/12R-TDEs/goodcode'
x_idx = readNPY(strcat(path, '/X_idx.npy'));
x_idxnew = readNPY(strcat(path, '/x_idxnew.npy'));
y_idx = readNPY(strcat(path, '/y_idx.npy'));
y_idxnew = readNPY(strcat(path, '/y_idxnew.npy'));
z_idx = readNPY(strcat(path, '/z_idx.npy'));
z_idxnew = readNPY(strcat(path, '/z_idxnew.npy'));
vol_idx = readNPY(strcat(path, '/vol_idx.npy'));
radden_idxnew = readNPY(strcat(path, '/radden_idxnew.npy'));

dx = 0.5 * vol_idx^(1/3)
Finterp = scatteredInterpolant(x_idxnew, y_idxnew, z_idxnew, radden_idxnew);
gradx_p = Finterp(x_idx+dx, y_idx , z_idx );
gradx_m = Finterp(x_idx-dx, y_idx, z_idx );
gradx = (gradx_p - gradx_m)./(2*dx);
plot(gradx)


