n = 128;
dt = 0.01;
timesteps = 100;
nu = 0.01;
eta = 0.02;
skip = 10;

butcher_rk4=[0, 0, 0, 0, 0; 0.5, 0.5, 0, 0, 0; 0.5, 0, 0.5, 0, 0; 1, 0, 0, 1, 0; 0, 1/6, 1/3, 1/3, 1/6];

omega = @(x, y) sin(x) .* sin(y);
A = @(x, y) cos(x) .* cos(y);
forcing = @(x, y) 0;

% known model:
% d(jB_x)/dx = 4cos(x)sin(x)cos(y)sin(y)
% d(jB_y)/dy = -4sin(x)cos(x)cos(y)sin(y)
% d(jB_x)/dx + d(jB_y)/dy = 0

[x, y, t, u, v, Bx, By] = mhd_rk_generator(omega, A, forcing, n, dt, timesteps, nu, eta, 1, skip, butcher_rk4, true);