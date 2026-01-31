clearvars; clc;
% NOTE:
% This script reproduces the methodology of a reference PINN implementation
% for 2D incompressible Navier–Stokes equations.
% Due to hardware and license constraints, full training may not converge locally.

% load simulation data
raw = load('cylinder_nektar_wake.mat');
X_star = raw.X_star;
t_star = raw.t;
U_star = raw.U_star;
P_star = raw.p_star;

N = size(X_star,1);            
Tt = numel(t_star);           

% make full coordinate and field vectors
x_full  = repmat(X_star(:,1), Tt, 1);
y_full  = repmat(X_star(:,2), Tt, 1);
t_full  = reshape(repmat(t_star', N, 1), N*Tt, 1);
u_full  = reshape(U_star(:,1,:), N*Tt, 1);
v_full  = reshape(U_star(:,2,:), N*Tt, 1);
p_full  = reshape(P_star,       N*Tt, 1);

% split into train and test
totalPoints = numel(x_full);
splitIdx = round(0.7 * totalPoints);
rng(0);
idxAll = randperm(totalPoints);

trainIdx = idxAll(1:splitIdx);
testIdx  = idxAll(splitIdx+1:end);

x_train = x_full(trainIdx); y_train = y_full(trainIdx); t_train = t_full(trainIdx);
u_train = u_full(trainIdx); v_train = v_full(trainIdx); p_train = p_full(trainIdx);

x_test = x_full(testIdx); y_test = y_full(testIdx); t_test = t_full(testIdx);
u_test = u_full(testIdx); v_test = v_full(testIdx); p_test = p_full(testIdx);

% normalization
mu_in  = mean([x_train,y_train,t_train],1);
s_in   = std([x_train,y_train,t_train],1);
mu_out = mean([u_train,v_train,p_train],1);
s_out  = std([u_train,v_train,p_train],1);

inputNorm.mean = mu_in; inputNorm.std = s_in;
outputNorm.mean = mu_out; outputNorm.std = s_out;

xN_train = (x_train-mu_in(1))/s_in(1);
yN_train = (y_train-mu_in(2))/s_in(2);
tN_train = (t_train-mu_in(3))/s_in(3);
uN_train = (u_train-mu_out(1))/s_out(1);
vN_train = (v_train-mu_out(2))/s_out(2);
pN_train = (p_train-mu_out(3))/s_out(3);

xN_test = (x_test-mu_in(1))/s_in(1);
yN_test = (y_test-mu_in(2))/s_in(2);
tN_test = (t_test-mu_in(3))/s_in(3);

% collocation points (REDUCED)
Nf = 2000;
minTrain = min([xN_train,yN_train,tN_train],[],1);
maxTrain = max([xN_train,yN_train,tN_train],[],1);
Xf_pool = minTrain + (maxTrain-minTrain).*rand(Nf,3);

% network (REDUCED SIZE)
layers = [3 64 64 3];
lgraph = layerGraph(featureInputLayer(3,'Normalization','none','Name','in'));
for i=2:length(layers)-1
    lgraph = addLayers(lgraph,[
        fullyConnectedLayer(layers(i),'Name',"fc"+(i-1))
        tanhLayer('Name',"tanh"+(i-1))]);
    if i==2
        lgraph = connectLayers(lgraph,'in',"fc1");
    else
        lgraph = connectLayers(lgraph,"tanh"+(i-2),"fc"+(i-1));
    end
end
lgraph = addLayers(lgraph,fullyConnectedLayer(3,'Name','out'));
lgraph = connectLayers(lgraph,"tanh"+(length(layers)-2),'out');
net = dlnetwork(lgraph);

viscosity = dlarray(0.01,'CB');

% TRAINING (REDUCED)
maxEpochs = 2;
miniBatchSize = 256;
learnRate = 1e-3;
trAvg=[]; trAvgSq=[]; iteration=0;

for epoch=1:maxEpochs
    idx = randperm(splitIdx);
    xN_train=xN_train(idx); yN_train=yN_train(idx); tN_train=tN_train(idx);
    uN_train=uN_train(idx); vN_train=vN_train(idx); pN_train=pN_train(idx);

    sub = randperm(Nf,300);   % REDUCED
    Xf_epoch = Xf_pool(sub,:);

    for i=1:miniBatchSize:splitIdx
        iteration = iteration+1;
        ib = i:min(i+miniBatchSize-1,splitIdx);

        dlX = dlarray([xN_train(ib)';yN_train(ib)';tN_train(ib)'],'CB');
        dlU = dlarray(uN_train(ib)','CB');
        dlV = dlarray(vN_train(ib)','CB');
        dlP = dlarray(pN_train(ib)','CB');

        cf = randperm(300,numel(ib));
        dlXf = dlarray(Xf_epoch(cf,:)','CB');

        [lossData,lossPhys,~,grads] = dlfeval(@modelGradients,net,viscosity,dlX,dlU,dlV,dlP,dlXf,inputNorm,outputNorm);
        [net,trAvg,trAvgSq] = adamupdate(net,grads.net,trAvg,trAvgSq,iteration,learnRate);
        viscosity = viscosity - learnRate*grads.viscosity;

        if mod(iteration,100)==0
            fprintf("Epoch %d Iter %d | Loss=%.3e | nu=%.4f\n",epoch,iteration,lossData,extractdata(viscosity));
        end
    end
end

fprintf("Adam done. Starting L-BFGS...\n");

% L-BFGS (REDUCED)
opts = optimoptions('fminunc','Algorithm','quasi-newton',...
    'SpecifyObjectiveGradient',true,'MaxIterations',5,'Display','iter');

params0.net=net; params0.viscosity=viscosity;
theta0 = double(gather(packNetworkAndViscosity(net,viscosity)));

theta = fminunc(@(th)objectiveLBFGS(th,params0,inputNorm,outputNorm,...
    xN_train,yN_train,tN_train,uN_train,vN_train,pN_train,Xf_pool),theta0,opts);

[net,viscosity] = unpackTheta(theta,params0);

save('trained_PINN_FAST.mat','net','viscosity','inputNorm','outputNorm','trainIdx','testIdx','X_star','t_star');
fprintf("Training complete. Saved model.\n");
