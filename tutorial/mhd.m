function [x,y,t,u,v,p] = mhd(n, dt, timesteps, nu, eta, skip)
  grid1d = (0:(n-1))/n*2*pi;
  [x,y] = meshgrid(grid1d);

  omega = 3*sin(3*x).*sin(3*y) + sin(x).*sin(y);
  
  % b field vector potential
  A = sin(2*x).*sin(2*y) + 0.5*sin(3*x).*sin(3*y);

  u = zeros(n,n,timesteps);
  v = zeros(n,n,timesteps);
  p = zeros(n,n,timesteps);
  
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;

  mask = abs(k) <= n/3;
  mask = mask & mask.';

  kx = k;
  ky = k.';

  to_u = 1i*ky ./(kx.^2 + ky.^2);
  to_v = -1i*kx./(kx.^2 + ky.^2);
  to_p = 1./(kx.^2 + ky.^2);

  to_p(1,1) = 0;
  to_u(1,1) = 0;
  to_v(1,1) = 0;
  
  to_u = to_u.*mask;
  to_v = to_v.*mask;
  to_p = to_p.*mask;

  kx = kx .*mask;
  ky = ky .*mask;

  omega = fft2(omega);
  A = fft2(A);

  e_omega = exp(-dt/2 * nu * (kx.^2 + ky.^2));
  e_A = exp(-dt/2 * eta * (kx.^2 + ky.^2));

  for t = 1:timesteps
    u(:,:,t) = real(ifft2(to_u.*omega));
    v(:,:,t) = real(ifft2(to_v.*omega));

    ux = real(ifft2(1i*kx.*to_u.*omega));
    uy = real(ifft2(1i*ky.*to_u.*omega));
    vx = real(ifft2(1i*kx.*to_v.*omega));
    vy = real(ifft2(1i*ky.*to_v.*omega));
    p(:,:,t) = real(ifft2(to_p.*fft2(ux.^2 + vy.^2 + 2*uy.*vx)));

    % coupled rk4
    for i = 1:skip
      [k1_omega, k1_A] = nonlinear_terms(omega, A, kx, ky, to_u, to_v, mask);
      k1_omega = dt * k1_omega;
      k1_A = dt * k1_A;

      omega = e_omega .* omega;
      A = e_A .* A;

      k1_omega = e_omega .* k1_omega;
      k1_A = e_A .* k1_A;

      [k2_omega, k2_A] = nonlinear_terms(omega + k1_omega/2, A + k1_A/2, kx, ky, to_u, to_v, mask);
      k2_omega = dt * k2_omega;
      k2_A = dt * k2_A;

      [k3_omega, k3_A] = nonlinear_terms(omega + k2_omega/2, A + k2_A/2, kx, ky, to_u, to_v, mask);
      k3_omega = dt * k3_omega;
      k3_A = dt * k3_A;

      omega = e_omega .* omega;
      A = e_A .* A;

      k2_omega = e_omega .* k2_omega;
      k3_omega = e_omega .* k3_omega;
      k2_A = e_A .* k2_A;
      k3_A = e_A .* k3_A;

      [k4_omega, k4_A] = nonlinear_terms(omega + k3_omega, A + k3_A, kx, ky, to_u, to_v, mask);
      k4_omega = dt * k4_omega;
      k4_A = dt * k4_A;
      k4_omega = e_omega .* k4_omega;
      k4_A = e_A .* k4_A;

      omega = omega + (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)/6;
      A = A + (k1_A + 2*k2_A + 2*k3_A + k4_A)/6;
    end

    imagesc(real(ifft2(omega)));
    axis square;
    colorbar();
    drawnow;
  end

  x = grid1d;
  y = grid1d;
  t = (0:timesteps-1)*dt*skip;
end

function [nl_omega, nl_A] = nonlinear_terms(omega, A, kx, ky, to_u, to_v, mask)
  u = real(ifft2(to_u .* omega));
  v = real(ifft2(to_v .* omega));
  
  Bx = real(ifft2(-1i * ky .* A));
  By = real(ifft2(1i * kx .* A));
  
  wx = real(ifft2(1i * kx .* omega));
  wy = real(ifft2(1i * ky .* omega));
  adv_omega = - (u .* wx + v .* wy);
  
  j = real(ifft2(-(kx.^2 + ky.^2) .* A));
  d_jx = real(ifft2(1i * kx .* (-(kx.^2 + ky.^2) .* A)));
  d_jy = real(ifft2(1i * ky .* (-(kx.^2 + ky.^2) .* A)));

  mhd_term = - (Bx .* d_jx + By .* d_jy);
  
  nl_omega = adv_omega + mhd_term;
  
  A_phys = real(ifft2(A));
  dAx = real(ifft2(1i * kx .* A));
  dAy = real(ifft2(1i * ky .* A));
  nl_A = - (u .* dAx + v .* dAy);
  
  nl_omega = mask .* fft2(nl_omega);
  nl_A = mask .* fft2(nl_A);
end