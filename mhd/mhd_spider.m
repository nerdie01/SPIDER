clear;
pkg load symbolic;

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

[x, y, t, u, v, p, Bx, By] = mhd_rk_generator(omega, A, forcing, n, dt, timesteps, nu, eta, 1, skip, butcher_rk4, true);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2: compute a library matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath("../SPIDER_functions/");
addpath("../tutorial/")

number_of_library_terms = 2;    %under-estimate this
number_of_windows       = 128;  %number of domains we integrate over 
degrees_of_freedom      = 1;    %scalars have one degree of freedom
dimension               = 3;    %how many dimensions does our data have?
envelope_power          = 4;    %weight is (1-x^2)^power
size_vec                = [64,64,32]; %how many gridpoints should we use per integration?
buffer                  = 0;    %Don't use points this close to boundary

% BEGIN BOILERPLATE
%define shorthand notation
nl = number_of_library_terms;
nw = number_of_windows;
dof= degrees_of_freedom;

%Make important objects for integration
pol      = envelope_pol( envelope_power, dimension );
G        = zeros( dof*nw, nl );
labels   = cell(nl, 1);
scales   = zeros(1,nl);

size_of_data = size(u, 1:dimension);
seed = 1;
corners = pick_subdomains_manual_seed( size_of_data, size_vec, buffer, nw, seed );

grid = { y,x,t };
a = 1; %running index over library
% END BOILERPLATE



labels{a} = "d_x u";
G(:,a)    = SPIDER_integrate( u, [2], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_y u";
G(:,a)    = SPIDER_integrate( u, [1], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_x v";
G(:,a)    = SPIDER_integrate( v, [2], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_y v";
G(:,a)    = SPIDER_integrate( v, [1], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_x p";
G(:,a)    = SPIDER_integrate( p, [2], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_y p";
G(:,a)    = SPIDER_integrate( p, [1], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_t u";
G(:,a)    = SPIDER_integrate( u, [3], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_t v";
G(:,a)    = SPIDER_integrate( v, [3], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_x Bx";
G(:,a)    = SPIDER_integrate( Bx, [2], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_y By";
G(:,a)    = SPIDER_integrate( By, [1], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_t Bx";
G(:,a)    = SPIDER_integrate( Bx, [3], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;

labels{a} = "d_t By";
G(:,a)    = SPIDER_integrate( By, [3], grid, corners, size_vec, pol );
scales(a) = 1;
a = a+1;


%normalize the feature matrix.
norm_vec = SPIDER_integrate( 0*u + 1, [], grid, corners, size_vec, pol );      
G        = G./norm_vec;
%Nondimensionalize
G = G./scales;









%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 3: sparse regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gamma = 1.25;

%search for three models
for i = 1:2
  %Do sparse regression
  [cs, residuals] = greedy_regression_pure_matlab( G );

  %Print the discovered model
  k = report_identified_model(cs, residuals, scales, labels, gamma);

  %remove the most important term
  [~, kill] = max( vecnorm(G*diag(cs(:,k))) );
  G(:, kill) = [];
  labels(kill) = [];
  scales(kill) = [];
end
