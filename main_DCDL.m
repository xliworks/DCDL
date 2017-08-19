%% DCDL--Gaussian dictionary beta version
% This matlab code implements the DCDL (Dictionary Learning with Difference of Convex programming) algorithm.
% 201703
% Xiang Li    xliworks@github

clear;
close all;
clc;

%% initialization
% penalty function: Minimax Concave Penalty (MCP)
lambda=0.1; gamma=50000; iter_max=1000; length_gain=30;% lambda>0   gamma>1

% cardinality, the number of non-zeros
cardi=1;  

% noise  
SNR=20;  % The modified parameters are possibly required when SNR is low.

% dictionary updating inneriiteration maximum. 
% If 1 is hard to reach good result, try increasing, for instance, 10.
inner_iter_max=1;

% dictionary updating tolerance
tol=1e-5;  

% numerical operation
epsilon=1e-6; 

% recovery Ratio and total error
recoveryRatio=zeros(iter_max,1);
totalErr=zeros(iter_max,1);

% generate ground truth dictionary
n=30;     % dim_row, Dictionary
k=50;     % dim_column, Dictionary
L=k*length_gain;    % dim_length, signal and coefficient
Dictionary_True=randn(n,k);

% normalize ground truth dictionary
Dictionary_True=Dictionary_True*diag(1./sqrt(sum(Dictionary_True.*Dictionary_True)));

% generate data
X_Coef_Matrix=zeros(k,L);
X_Coef_Matrix(1:cardi,:)=randn(cardi,L);
for i=1:L
    X_Coef_Matrix(:,i)=X_Coef_Matrix(randperm(k),i);
end
Y_Data_Matrix=Dictionary_True*X_Coef_Matrix;

% observation noise   
if SNR~=Inf
    stdnoise = std(reshape(Y_Data_Matrix,n*L,1))*10^(-SNR/20);
    noise = randn(n,L) * stdnoise;
    % noisy signal
    Y_Data_Matrix = Y_Data_Matrix + noise;
end

% initialize dictionary
Dictionary=randn(n,k);
Dictionary=Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));

% initialize coefficients
X_Coef_Matrix=zeros(k,L);
X_Coef_Matrix(1:cardi,:)=randn(cardi,L);
for i=1:L
   X_Coef_Matrix(:,i)=X_Coef_Matrix(randperm(k),i);
end

X_Coef_abs = zeros(size(X_Coef_Matrix));
Z=zeros(k,L);

%% main iteration
for iter=1:iter_max
    %% Sparse coding phase
    X_Coef_Matrix_former=X_Coef_Matrix;
            
    % penalty MCP
    X_Coef_abs = abs(X_Coef_Matrix_former);
    ind = find(X_Coef_abs > lambda*gamma) ;
    Z(ind)=lambda*sign(X_Coef_Matrix_former(ind));
    ind = find(X_Coef_abs <= lambda*gamma) ;
    Z(ind)=X_Coef_Matrix_former(ind)./gamma;
               
    Dt=Dictionary';
    Dsq=Dt*Dictionary;
    phi=norm(Dsq); 
            
    Grad=zeros(size(X_Coef_Matrix));
    for ii=1:L
       Grad(:,ii) = Dt*(Dictionary*X_Coef_Matrix_former(:,ii)-Y_Data_Matrix(:,ii));
    end
%      Grad = Dt*(Dictionary*X_Coef_Matrix_former-Y_Data_Matrix);

    u_update = X_Coef_Matrix_former-Grad./phi;

    X_Coef_Matrix = sign(u_update).*max(abs(u_update+(1/phi).*Z)-lambda/phi,0);
        
    %% dictionary updating phase

    A=X_Coef_Matrix*X_Coef_Matrix';
    B=Y_Data_Matrix*X_Coef_Matrix';

    omega=abs(A)*ones(k,1);
    omega=max(omega,epsilon);
    Omega_Matrix=zeros(n,k);
    for i=1:n
       Omega_Matrix(i,:)=omega';
    end

    % inner iteration
    for inner_iter=1:inner_iter_max
       Dictionary_former=Dictionary;
       Dictionary_hat=Omega_Matrix.*Dictionary-(Dictionary*A-B);

       for j=1:k
         Dictionary(:,j)=Dictionary_hat(:,j)/max(omega(j),norm(Dictionary_hat(:,j)));
       end

       Dictionary=Dictionary*diag(1./sqrt(sum(Dictionary.*Dictionary)));
                
       if norm(Dictionary_former-Dictionary) < tol
         break 
       end
    end
    
    %% evaluation
    recoveryRatio(iter)=DictionariesDistance(Dictionary_True,Dictionary);   
    totalErr(iter)=sqrt(sum(sum((Y_Data_Matrix-Dictionary*X_Coef_Matrix).^2))/numel(Y_Data_Matrix));
    fprintf('iter=%d / %d  totalErr=%f recoveryRatio=%f \n',iter,iter_max,totalErr(iter),recoveryRatio(iter))

end

%% plot
figure(1);
subplot(211);
plot(recoveryRatio);
axis([0 iter_max 0 100]);
xlabel('Iteration');
ylabel('Recovery Ratio');
title('Recovery Ratio');
subplot(212);
plot(totalErr);
xlabel('Iteration');
ylabel('Error');
title('Total Err');
