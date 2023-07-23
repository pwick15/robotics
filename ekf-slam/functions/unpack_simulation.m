function unpacked_measurements = unpack_simulation(measurements)
    % assumes a 2 * m matrix where the m columns are to be stacked
    % vertically
    [~, cols] = size(measurements);
    unpacked_measurements = zeros(2*cols,1);
    for c = 1:cols
        unpacked_measurements(2*c-1: 2*c) = measurements(:,c);
    end
end