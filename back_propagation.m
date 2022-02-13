function [W, k, SSE_new, x_min, x_max, y_min, y_max] = back_propagation(x, y, tol, H, steps)
    % this function use back propagation in order to train the neural network
    n = length(x(:, 1)); % number of rows
    N = length(x(1, :));
    % normalizing the input
    x_min = min(x);
    x_max = max(x);
    x = (x-x_min) ./ (x_max-x_min);
    % normalising the output
    y_min = min(y);
    y_max = max(y);
    y = (y-y_min) ./ (y_max-y_min);
    
    % bias node in the input layer
    X = [ones(n, 1) x]; 
    % random starting values 
    W = unifrnd (0, 0.5, N+1, H);
    % the bias node in the hidden layer will be the last (H+1th)
    k = unifrnd (0, 0.5, H+1, 1);
    sigma = @(u) (1 + exp(-u)).^( -1);

    for m = 1:steps 
        mu = 10000 / (200000+m);
        % Compute v_star and then v using the activation function
        v_star = X*W;
        v = [sigma(v_star) ones(n, 1)]; % Compute the output 
        o = v*k;
        % Update weights w_ih 
        
        W = W + 2*mu*((X.*(y-o))'*(v(:, 1:H).*(1-v(:, 1:H)))).*k(1:H)';
        k = k + 2*mu*v'*(y-o);
        %Compute SSE 
        SSE_new(m) = ((norm(y-o, 2))^2); 
        if SSE_new(m) < tol 
            break
        end 
    end
end