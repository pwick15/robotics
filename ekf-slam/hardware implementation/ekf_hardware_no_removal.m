close all;

% PARAMETER INITIALISATION
x_i = 0; y_i = 0; theta_i = 0;
u = 0; q = 0;
counter = 1; dt = 0.5; SAMPLES = 5000; q_weight = 0.35; no_line = 0; total_time = 0; TIME_LIMIT = 1000;


% Add the ARUCO detector
addpath("/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/arucoDetector")
addpath("/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/arucoDetector/include")
addpath("/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/arucoDetector/dictionary")

% load necessary files and parameters
load('/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/calibration/calibratedCameraParams.mat');
load("/Users/punjayawickramasinghe/Documents/University/2022/Semester2/robotics/Labs/ENGNX627_code/lab4slam/lab4_code/arucoDetector/dictionary/arucoDict.mat");
marker_length = 0.070;


% All things piBot related ... 
pb = PiBot('192.168.50.1'); %Change to your appropriate address
figure(1);
camAxes = axes();


states = zeros(SAMPLES,2);
data = cell(SAMPLES,5);

% Initialise your EKF class
obj = slam_no_removal();
obj.x = [x_i;y_i; theta_i];


tic;

figure(2);
while true
    % First, get the current camera frame
    img = pb.getImage();
    [numRows,~] = size(img);
    croppedImg = img(numRows - 40: numRows, :,:);
    gray_img = rgb2gray(croppedImg);
    bin_img = ~imbinarize(gray_img, 0.3);
    figure(1)
    imshow(bin_img)
%     imshow(bin_img, 'Parent', camAxes);
    
     % Find the centre of the line to follow
    [r, c] = find(bin_img == 1);
    centre_of_mass = [mean(c), mean(r)];
    centre_of_mass = (centre_of_mass - [160, 20]) ./ [160, 20];
    
    
    % If x is negative, spin left. If x is positive, spin right
    q = -q_weight*centre_of_mass(1);
    
    % Drive forward as soon as the dot is roughly in view
    u = 0.06;

%     u = 0.08*(1-centre_of_mass(1)^2);
%     q = -0.6*centre_of_mass(1);

    
    % Compute the required wheel velocities
    [wl, wr] = inverse_kinematics(u,q);
    
    if abs(wl)<5
        wl = 5*(wl/abs(wl));
    end
    if abs(wr)<5
        wr = 5*(wr/abs(wr));
    end
    
    % Check the dot is visible anywhere
    if sum(bin_img,'all') < 42
        no_line = no_line + 1;
    else
        no_line = 0;
    end
    
    
    if no_line > 3
        dt = toc;
        pb.stop();
        no_line = 0;
        flag = 0;
        tp = counter
        while flag == 0
            
            img = pb.getImage();
            [numRows,~] = size(img);
            croppedImg = img(numRows - 40: numRows, :,:);
            gray_img = rgb2gray(croppedImg);
            bin_img = ~imbinarize(gray_img, 0.3);
            figure(1)
            imshow(bin_img)
            if sum(bin_img,'all') > 42
                tic;
                flag = 1;
            else
                pb.setVelocity([10,-10],1)
                total_time = total_time + 1;
                obj.x(3) = obj.x(3) + 0.8180138217;
            end
        end
    end
    
    dt = toc;

    % Apply the wheel velocities
    pb.setVelocity(wl,wr);
    tic;
    
    
    % detect landmarks
    [ids, measurements, ~] = detectArucoPoses(img, marker_length, cameraParams, arucoDict);
    
    if ~isempty(measurements)
        measurements = unpack_hardware(measurements);
    end
    
    new_ids = [];
    new_measurements = [];
    id_count = 1;
    
    for c = 1:length(ids)
        if ids(c) < 20
            new_ids(id_count) = ids(c);
            new_measurements(2*id_count-1:2*id_count) = measurements(2*c-1:2*c);
            id_count = id_count + 1;
        end  
    end
    
    ids = new_ids;
    measurements = new_measurements';
    
    % START EKF
    obj.input_velocity(dt, u, q);
    obj.input_measurements(ids, measurements);
    [state, state_cov] = obj.output_robot();
    [landmarks, landmark_cov] = obj.output_landmarks();
    landmarks = pack(landmarks);
    % END EKF 
    
    % store state
    states(counter, 1) = state(1);
    states(counter, 2) = state(2);
    
    
    data{counter, 1} = u;
    data{counter, 2} = q;
    data{counter, 3} = dt;
    data{counter, 4} = measurements;
    data{counter, 5} = ids;
    
    % live visualisation
    figure(2);
    xlim([-0.5 5.5])
    ylim([-0.5 5.5])
    plot(states(1:counter, 1), states(1:counter, 2), 'b');
    hold on;
    scatter(landmarks(:,1), landmarks(:,2), 4, 'm');
    hold on;
    
    % update counters
    counter = counter + 1;
    total_time = total_time + dt;
    
    if total_time > TIME_LIMIT % if 3 minutes have been exceeded
        break
    end
end

pb.stop();
hold off; 


% visualise

figure;
scatter(states(:,1),states(:,2)); 
xlabel('x','FontSize', 16);
ylabel('y','FontSize', 16);
title('EKF SLAM', 'FontSize', 20);
hold on; 

[landmarks, ~] = obj.output_landmarks();


for c = 1:length(landmarks)/2
    id = obj.idx2num(c);
    p1 = [landmarks(2*c-1) landmarks(2*c)];
    text(p1(1)+0.05,p1(2)+0.05, string(id), 'FontSize', 12)
    hold on;
    scatter(landmarks(2*c-1), landmarks(2*c), 40,'filled','r');
    hold on;
end
% 
convertToTXT(obj,1)
