function unpacked_measurements = unpack_hardware(measurements)
    % assumes a m x 3 matrix where the 3 columns are depth, width and height respectively
    % and the m rows of depth and height values are to be stacked into a
    % m x 1 matrix
    % vertically
    
    [rows, ~] = size(measurements);
    unpacked_measurements = zeros(2*rows,1);
    for c = 1:rows
        unpacked_measurements(2*c-1: 2*c) = measurements(c,1:2)';
    end
end