rmse_sum = 0;

for i = 1:IDs
   rmse_sum = rmse_sum + (norm(pi_hats_final(:,i) - landmark_true(:,i)))^2; % rmse
end

rmse = sqrt(1/IDs * rmse_sum);
