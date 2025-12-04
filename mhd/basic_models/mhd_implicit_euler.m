function [x,y,t,u,v,p,Bx,By] = mhd_implicit_euler(n, dt, timesteps, nu, eta, lorentz, skip, vis)
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
            [nl_omega, nl_A] = mhd_terms(omega, A, kx, ky, to_u, to_v, nu, eta, lorentz, mask);
            omega = omega + dt * nl_omega;
            A = A + dt * nl_A;
        end

        if vis
            visualize_fields(omega, A, "velocity", "B field")
        end
    end

    x = grid1d;
    y = grid1d;
    t = (0:timesteps-1)*dt*skip;
end