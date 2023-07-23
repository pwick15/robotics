

mu_hat_sum = [0;0];
mu_sum = [0;0];
Sigma_sum = [0,0;0,0];

for i = 1:IDs
   mu_hat_sum = mu_hat_sum + pi_hats_final(:,i); % u_hat sum
   mu_sum = mu_sum + landmark_true(:,i); % u sum
end
mu_hat = 1/IDs * mu_hat_sum;
mu = 1/IDs * mu_sum;

for i = 1:IDs
   Sigma_sum = Sigma_sum + (landmark_true(:,i) - mu)*(pi_hats_final(:,i) - mu_hat)';
end

Sigma = 1/IDs * Sigma_sum;


[U,D,V] = svd(Sigma);

if det(Sigma) >= 0
    A = [1, 0; 0 1];
else
    A = [1, 0; 0, -1];
end

R_S = V * A * U';
x_S = mu_hat - R_S * mu;


aRMSE_sum = 0;
for i = 1:IDs
    aRMSE_sum = aRMSE_sum + norm(R_S'*(pi_hats_final(:,i) - x_S) - landmark_true(:,i))^2;
end

aRMSE = sqrt(1/IDs * aRMSE_sum)