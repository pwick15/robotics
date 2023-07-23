function size = matrixMagnitude(A)
    size = sqrt(sum(A .* A,'all'));
end