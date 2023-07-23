function convertToTXT(obj)
    % assume that the obj is ekf object which contains 
    % the landmark estimates from the state vector
    % i.e. it will be a (2xN) vector containing the x and y positions of
    % lmks
    
    [landmarks, ~] = output_landmarks(obj);
    num_landmarks = length(landmarks)/2;
    ids = obj.idx2num';
    x = zeros(num_landmarks,1);
    y = zeros(num_landmarks,1);
    for c = 1:num_landmarks
        x(c) = landmarks(2*c - 1);
        y(c) = landmarks(2*c);
    end
    arr = zeros(length(landmarks)/2, 3);
    arr(:,1) = ids;
    arr(:,2) = x;
    arr(:,3) = y;
    writematrix(arr,'data.csv') 
    
%     T = table(ids, x, y);
% %     writetable(T,'ekf_G5_trial_'+string(trial_num)+'.txt');    
    
end