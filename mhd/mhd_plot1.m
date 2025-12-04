n = 128;
dt = 0.01;
timesteps = 50;
nu = 0.01;
eta = 0.02;
skip = 10;

c = 64

butcher_rk1 = [0, 0; 0, 1];
butcher_rk2 = [0, 0, 0; 0.5, 0.5, 0; 0, 0, 1];
butcher_rk3= [0, 0, 0, 0; 1/3, 1/3, 0, 0; 2/3, 0, 2/3, 0; 0, 1/4, 0, 3/4];
butcher_rk4=[0, 0, 0, 0, 0; 0.5, 0.5, 0, 0, 0; 0.5, 0, 0.5, 0, 0; 1, 0, 0, 1, 0; 0, 1/6, 1/3, 1/3, 1/6];
butcher_rk6=[0, 0, 0, 0, 0, 0, 0; 0.25, 0.25, 0, 0, 0, 0, 0; 3/8, 3/32, 9/32, 0, 0, 0, 0; 12/13, 1932/2197, -7200/2197, 7296/2197, 0, 0, 0; 1, 439/216, -8, 3680/513, -845/4104, 0, 0; 0.5, -8/27, 2, -3544/2565, 1859/4104, -11/40, 0; 0, 16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55];

figure;
hold on;

schemes={'RK1', 'RK2', 'RK3', 'RK4', 'RK6'};

butchers={butcher_rk1, butcher_rk2, butcher_rk3, butcher_rk4, butcher_rk6};

for i = 1:5
    [x, y, t, u, v, p, Bx, By] = mhd_rk_generator(n, dt, timesteps, nu, eta, 1, skip, butchers{i});
    u_point = squeeze(u(c, c, :));
    v_point = squeeze(v(c, c, :));
    mag = sqrt(u_point.^2+v_point.^2);
    plot(t, mag, 'DisplayName', schemes{i});
end

legend('show');
xlabel('time');
ylabel('velocity at center');
hold off;
