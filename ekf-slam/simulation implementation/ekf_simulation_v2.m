
addpath("/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/simulator");

% Loading collected data
% load('/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab3visualodometry/collected_data.mat');
% load('/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/lab4slam_v2/data_slam_v2.mat')
% load('/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab3visualodometry/collected_data_more_lmk.mat')
load('/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab3visualodometry/data_surrounding_lmks.mat');
data = table2cell(table);
true_xs = cell2mat(data(:,1));
true_ys = cell2mat(data(:,2));


% GENERAL INITIALISATION  
dt = 0.1;
SAMPLES = 1000;
IDs = 16;
[lmx,lmy] = meshgrid(0.5:(4/3):4.5);
landmark_true = [lmx(:)'; lmy(:)'];
states = zeros(3,SAMPLES); % storage of the ekf positions
pi_hats = zeros(3, IDs); % most up to date pi_hat
landmarks = zeros(SAMPLES,IDs * 2); % storage of all landmark positions in the inertial frame as they are observed
sample_numbers = zeros(IDs, 1);


% INITIALISE THE EKF
obj = ekf_slam_v2();
% FOR THE FIRST LOOP, VARIABLES THAT NEED TO BE PREDEFINED
obj.x = [1.5; 1.5; pi()/2];



% ----------- START VISUAL ODOMETRY ALGORITHM -------------------

for counter = 1:length(true_xs)

    % could remove these top three lines and do it outside the loop
    u = cell2mat(data(counter, 4));
    q = cell2mat(data(counter, 5));
    dt = cell2mat(data(counter, 6));
    z = unpack(cell2mat(data(counter, 7)));
    ids = cell2mat(data(counter, 8))';


    % assumes measurements is 2*id
    obj.input_velocity(dt, u, q);
    obj.input_measurements(ids, z);

    % -------------------- end EKF -----------------------------

    % store the position estimate
    [state, ~] = obj.output_robot();
    states(1:2, counter) = state(1:2);

    % store the landmark estimate
    for i = 1:length(ids)
        curr_id = ids(i);
        idx_lx = find(obj.idx2num == curr_id)*2 + 2;
        idx_ly = find(obj.idx2num == curr_id)*2 + 3;
        sto_idx_lx = idx_lx - 3;
        sto_idx_ly = idx_ly - 3;
        landmarks(counter, sto_idx_lx) = obj.x(idx_lx);
        landmarks(counter, sto_idx_ly) = obj.x(idx_ly);
    end
    [~, cov] = obj.output_robot;
    a = matrixMagnitude(cov);
    [~, cov] = output_landmarks(obj);
    b = matrixMagnitude(cov);
    [a,b]
%     if counter > 20
%         break;
%     end

end

figure;
scatter(states(1,:), states(2,:))
hold on;
scatter(true_xs, true_ys);
hold on;
scatter(landmark_true(1,:), landmark_true(2,:), 'k', 'filled');
hold on;
for i = 1:IDs
    scatter(landmarks(:,i*2-1), landmarks(:,i*2));
    hold on;
end


