function [wl,wr,u,q,stop] = lineFollow(img,q_weight,camAxes,threshold)

    stop = 0;
       
    [numRows,~] = size(img);
    img = img(numRows - 29: numRows, :,:);
    gray_img = rgb2gray(img);
    bin_img = ~imbinarize(gray_img, threshold);
    imshow(bin_img, 'Parent', camAxes);
    

    % Find the centre of the line to follow
    [r, c] = find(bin_img == 1);
    centre_of_mass = [mean(c), mean(r)];
    centre_of_mass = (centre_of_mass - [200, 15]) ./ [200, 15];
    
    % Check the dot is visible anywhere
    if ~any(bin_img)
        stop = 1;
    end
    
    % If x is negative, spin left. If x is positive, spin right
    q = -q_weight*centre_of_mass(1);
    u = 0.1;
   
    % Compute the required wheel velocities
    [wl, wr] = inverse_kinematics(u,q);
    
    if abs(wl)<5
        wl = 5*(wl/abs(wl));
    end
    if abs(wr)<5
        wr = 5*(wr/abs(wr));
    end

end