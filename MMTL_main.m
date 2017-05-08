
%% P is the final projection 

%% A1 is the one modelity of Source dataset, A2 is another modality
%% B1 is the one modelity of Tagret dataset, B2 is another modality

%% A1, A2, B1, B2 data matrix (sample size x dim)
Xs = [A1', B1'];
Xt = B2';

ma = mean(A1,1);
mb = mean(B1,1);
mab = ma - mb;

options.ReducedDim = 100;

%% initialize subspace
[P,~] = PCA(Xs', options);
Ys = P'*Xs;

%% initialize variables
d = size(Xt,1);
m = size(Ys,2);
p = size(Xt,2);

np = size(P,2);

Z = zeros(p,m);
J = zeros(p,m);

L = zeros(np,np);
S = zeros(np,np);

E = zeros(np,m);
Q = zeros(size(P));
Y3 = zeros(np,m);
Y1 = zeros(p,m);
Y2 = zeros(np,np);
Y4 = zeros(np,d);

max_mu = 1e8;
rho = 1.3;
mu = 1e-6;
rsa = [];

alpha = 0.1;
lambda = 0.01;
tol = 1e-8;
beta = 0.2;
maxIter = 100;
iter = 0;
rol = [];
while iter < maxIter
    iter = iter + 1;
    %%update Q
    temp = P'+Y4/mu;
    Q = solve_l1l2(temp,alpha/mu);
    %%update P
    if iter>1
        P1 = 2*beta*mab'*mab + eye(d) + Xt*Z*Z'*Xt';
        P2 = Q + (Ys - L*Ys - E)*Z'*Xt' - (Y4 - Y3*Z'*Xt')/mu;
        P = (P2/P1)';
        P = orth(P);
    end

    %%update J
    temp = Z + Y1/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    J = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    rZ = svp; %rank(J);
    
    
    %%update S
    temp = L + Y2/mu;
    [U,sigma,V] = svd(temp,'econ');
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    S = U(:,1:svp)*diag(sigma)*V(:,1:svp)';
    rL = svp; %rank(S);   
    
    %%update Z
    Z1 =  eye(p) + Xt'*P*P'*Xt;
    Z2 =  Xt'*P*(Ys - L*Ys - E) + J + (Xt'*P*Y3 - Y1)/mu;
    Z = Z1\Z2;
    
    %%update L
    L1 = eye(np)+Ys*Ys';
    L2 = S + (Ys - P'*Xt*Z - E)*Ys' + (Y3*Ys' - Y2)/mu;
    L = L2/L1;
    
    %%update E
    tmp_E = Ys -P'*Xt*Z-L*Ys+Y3/mu;
    E = max(0,tmp_E - lambda/mu)+min(0,tmp_E + lambda/mu);
    
      
    %%update the multiplies
    leq1 = Ys -P'*Xt*Z-L*Ys-E;
    leq2 = Z-J;
    leq3 = L-S;
    leq4 = P'-Q;
    stopC = max(max(max(abs(leq1))),max(max(abs(leq2))));
    stopC = max(max(max(abs(leq3))),stopC);
    stopC = max(max(max(abs(leq4))),stopC);
    if stopC<tol 
        break;
    else
        Y1 = Y1 + mu*leq2;
        Y2 = Y2 + mu*leq3;
        Y3 = Y3 + mu*leq1;
        Y4 = Y4 + mu*leq4;
        mu = min(max_mu,mu*rho);
    end
end
