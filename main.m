%setup
set.seed = 1;
N0 = 1000; 
S0 = N0-1;
I0 = 1;
% value of p used for computing H(p)
p = [0.003, 0.00275, 0.0025, 0.00225, 0.002, ... 
    0.00175, 0.0015, 0.00125, 0.001, 0.00075, 0.0005];
p = sort(p);
lp = length(p);
simulations = 1000;
days = 60;
cp = @(x) ((1000 / 3) * x).^( -9) - 1;
sim_p_mean = zeros(days + 1, lp);
sim_p_var = zeros(days + 1, lp);

% color gradient
c2 = [1, 0, 0];
c1 = [1, 0.8, 0];
colors_p = [linspace(c1(1), c2(1), lp)', linspace(c1(2), c2(2), lp)', ...
    linspace(c1(3), c2(3), lp)'];

% simulate pandemic evolutions for each value of p
for i = 1:lp
    [sim_p_mean(:, i), sim_p_var(:, i)] = sim(S0, I0, p(i), simulations, days);
end
figure('Position', [10 10 1000 330]) % plot of avg pandemics
hold on
for i = 1:lp
    plot(1:days + 1, sim_p_mean(:, i), "LineWidth", 2, "Color", colors_p(i, :)) 
    axis ([0 61 0 inf])
end
hold off
saveas(gcf, 'AverageEvolutions.png')
figure('Position', [10 10 1000 330]) % plot of some pandemics with respective sd
hold on
for i = 1:lp
    if i == 11 || i == 7 || i == 5
        plot(1:days + 1, sim_p_mean(:, i), "LineWidth", 2, "Color", colors_p(i, :)) 
        axis ([0 61 0 inf])
        
        sd = sqrt(sim_p_var(:, i));
        curve1 = max(zeros(days + 1, 1), sim_p_mean(:, i) - sd);
        curve2 = sim_p_mean(:, i) + sd;
        x=(1:days + 1)';
        x2 = [x; fliplr(x)];
        inBetween = [curve1;  fliplr(curve2)];
        fill(x2, inBetween, colors_p(i,:), 'FaceAlpha', 0.5, "LineStyle", "none");
    end
end
hold off
saveas(gcf, 'sdEvolutions.png')
figure('Position', [10 10 1000 330]) % plot of costs
slg = semilogy(p , cp(p),'-o', p , sum(sim_p_mean, 1) , '-o', p, ...
    cp(p) + sum(sim_p_mean, 1), '-o', "LineWidth", 2);
slg(1).Color = [1, 0, 0];
slg(2).Color = [1, 0.6, 0];
slg(3).Color = [0.9, 0.9, 0];
legend ({'c(p)', 'H(p)', 'C(p)'}) 
axis ([0.0005 0.003 0 inf])
saveas(gcf,'costs.png')

% training the neural network
x = p(4: end)';
y = (cp(p(4:end)) + sum(sim_p_mean(:, 4:end), 1))';
tol = 1e-5;
H = 10;
steps = 1000000;
[W, k, SSE_new, x_min, x_max, y_min, y_max] = back_propagation(x, y, tol, H, steps);
xx = 0.00125:0.00005:0.003;
yy = zeros(1, length(xx));
for i = 1:length(xx)
    yy(i) = evalNN(xx(i), W, k, x_min, x_max, y_min, y_max);
end
figure('Position', [10 10 1000 330]) % plot of neural network estimate
hold on
plot(xx, yy, 'Color', [1, 0.6, 0], "LineWidth", 2);
plot(x, y, '.r', 'MarkerSize', 24)
legend ({'Predicted C(p)','Known C(p) values'}) 
hold off
saveas(gcf, 'nn.png')

% setup of steepest descend
x0 = 2e-3;
f = @(x) evalNN(x, W, k, x_min, x_max, y_min, y_max);
gradf = @(x) (f(x + 1e-8) - f(x - 1e-8)) / 2e-8;
alpha0 = 1e-8;
kmax = 10000;
c1 = 1e-4;
rho = 0.8;
btmax = 50;
tolgrad = 1e-8;
[xk, fk, gradfk_norm, kk, xseq, btseq] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha0, kmax, tolgrad, c1, rho, btmax);
fseq = zeros(1, length(xseq));
for i = 1:length(xseq)
    fseq(i) = f(xseq(i));
end
figure('Position', [10 10 1000 330]) % plot of the estimated minimum
hold on
plot(xx, yy, 'Color', [1, 0.6, 0], "LineWidth", 2);
plot(x, y, '.r', 'MarkerSize', 24);
plot(xseq, fseq, '.','Color', [0.5 0.5 0.5], 'MarkerSize', 16);
plot(xk, fk, '.k', 'MarkerSize', 32);
legend ({'Predicted C(p)', 'Known C(p) values', ...
    'Steepest Descent steps', 'Approximated minimum'}) 
hold off
saveas(gcf, 'minimum.png')

function [meanNewInfected, varNewInfected] = sim(S0, I0, p, simulations, days)
    % this function is responsible to estimate the avg evolution of the
    % pandemic, at fixed p
    meanNewInfected = ones(days + 1, 1);
    varNewInfected = zeros(days + 1, 1);

    for s = 1:simulations
        I = I0;
        S = S0;
        for d = 2:days + 1
            newInfected = binornd(S, 1 - (1-p)^(I)); 
            S = S - newInfected; 
            I = newInfected; 
            if s > 1
                varNewInfected(d) = (1 - 1/(s-1))*varNewInfected(d) + ...
                    1/s*(newInfected-meanNewInfected(d))^2;
            end
            meanNewInfected(d) = (1 - 1/s)*meanNewInfected(d) + 1/s*newInfected;
        end
    end
end