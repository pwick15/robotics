function output = pack(input)
    output = zeros(length(input)/2,2);
    for c = 1:length(input)/2
        lx = input(2*c-1);
        ly = input(2*c);
        output(c,:) = [lx,ly];
    end
end