function [x,y,t,u,v,p,Bx,By] = mhd_rk4(n, dt, timesteps, nu, eta, lorentz, skip, vis)
    if nargin <8 || isempty(vis)
        vis = false
    end

    grid1d = (0:(n-1))/n*2*pi;
    [x, y] = meshgrid(grid1d);

    omega = 3*sin(3*x).*sin(3*y) + sin(x).*sin(y);
    A = sin(x).*sin(y);

    u = zeros(n, n, timesteps);
    v = zeros(n, n, timesteps);
    p = zeros(n, n, timesteps);
    Bx = zeros(n, n, timesteps);
    By = zeros(n, n, timesteps);

    k = 0:n-1;
    k(k>n/2) = k(k>n/2) - n;

    mask = abs(k) <= n/3;
    mask = mask & mask.';

    kx = k;
    ky = k.';

    kk = kx.^2 + ky.^2;
    kk(1,1) = 1;

    to_u = 1i*ky ./ kk;
    to_v = -1i*kx ./ kk;
    to_p = 1 ./ kk;

    to_u(1,1) = 0;
    to_v(1,1) = 0;
    to_p(1,1) = 0;

    to_u = to_u .* mask;
    to_v = to_v .* mask;
    to_p = to_p .* mask;

    kx = kx .* mask;
    ky = ky .* mask;

    omega = fft2(omega);
    A = fft2(A);

    for t = 1:timesteps
        u(:,:,t) = real(ifft2(to_u .* omega));
        v(:,:,t) = real(ifft2(to_v .* omega));
        Bx(:,:,t) = real(ifft2(1i*ky .* A));
        By(:,:,t) = real(ifft2(-1i*kx .* A));

        ux = real(ifft2(1i*kx.*to_u.*omega));
        uy = real(ifft2(1i*ky.*to_u.*omega));
        vx = real(ifft2(1i*kx.*to_v.*omega));
        vy = real(ifft2(1i*ky.*to_v.*omega));
        p(:,:,t) = real(ifft2( to_p.*fft2( ux.^2 + vy.^2 + 2*uy.*vx ) ));

        for i = 1:skip
            [k1_omega, k1_A] = mhd_terms(omega, A, kx, ky, to_u, to_v, nu, eta, lorentz, mask);
            k1_omega = dt * k1_omega;
            k1_A = dt * k1_A;

            [k2_omega, k2_A] = mhd_terms(omega + k1_omega/2, A + k1_A/2, kx, ky, to_u, to_v, nu, eta, lorentz, mask);
            k2_omega = dt * k2_omega;
            k2_A = dt * k2_A;

            [k3_omega, k3_A] = mhd_terms(omega + k2_omega/2, A + k2_A/2, kx, ky, to_u, to_v, nu, eta, lorentz, mask);
            k3_omega = dt * k3_omega;
            k3_A = dt * k3_A;

            [k4_omega, k4_A] = mhd_terms(omega + k3_omega, A + k3_A, kx, ky, to_u, to_v, nu, eta, lorentz, mask);
            k4_omega = dt * k4_omega;
            k4_A = dt * k4_A;

            omega = omega + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)/6;
            A = A + (k1_A + 2*k2_A + 2*k3_A + k4_A)/6;
        end

        if vis
            visualize_fields(omega, A, "velocity", "B field")
        end
    end

    x = grid1d;
    y = grid1d;
    t = (0:timesteps-1)*dt*skip;
end