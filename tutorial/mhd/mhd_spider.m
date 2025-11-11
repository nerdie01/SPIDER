n = 128;
dt = 0.01;
timesteps = 1000;
nu = 0.01;
eta = 0.02;
skip = 10;

[x, y, t, u, v, Bx, By] = mhd_implicit_euler(n, dt, timesteps, nu, eta, 1, skip, true);
