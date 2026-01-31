clear; clc;

% Load data
raw = load('../data/cylinder_nektar_wake.mat');
X = raw.X_star;
t = raw.t;
U = raw.U_star;
P = raw.p_star;

% Pick one time snapshot
tid = 10;
x = X(:,1); 
y = X(:,2);
u = U(:,1,tid);
v = U(:,2,tid);
p = P(:,tid);

% Velocity magnitude
speed = sqrt(u.^2 + v.^2);

figure;
scatter(x,y,15,speed,'filled');
colorbar;
title('Velocity magnitude (sample snapshot)');
xlabel('x'); ylabel('y');

fprintf('Demo run successful. Data loaded and visualized.\n');
