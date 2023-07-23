function landmarks = placeRandomLandmarks(numLmks, seed)
    rng(seed);
    landmarks = 5*rand(numLmks,2)';
end