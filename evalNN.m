function o = evalNN(x, W, k, x_min, x_max, y_min, y_max)
    % this function evaluate the output of a neural network
    % x is a row vector
    sigma = @(x) (1 + exp(-x)).^(-1);
    x = (x-x_min) / (x_max-x_min); 
    v = sigma([1 x]*W);
    o = [v ones(length(v(:, 1)), 1)]*k;
    o = (y_max-y_min)*o + y_min;
end