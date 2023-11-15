clear; clc;

% Load and standardise data
data = csvread('iris-data.csv');
data = data / max(data(:));
labels = csvread('iris-labels.csv');

numberOfPatterns = size(data,1);
numberOfFeatures = size(data,2);

% Initialize weights from uniform distribution [0,1]
outputSize = [40, 40, numberOfFeatures];
weights = rand(outputSize);
initialWeights = weights;

% Parameters
eta_0 = 0.1;
d_eta = 0.01;
sigma_0 = 10;
d_sigma = 0.05;

% Training loop
numberOfEpochs = 10;
for epoch = 1:numberOfEpochs
    eta = eta_0 * exp(-d_eta * epoch);
    sigma = sigma_0 * exp(-d_sigma * epoch);

    for p = 1:numberOfPatterns
        x = data(p, :);
        
        % Calculate distances
        distances = sum((reshape(weights, [], numberOfFeatures) - x).^2, 2);
        [~, winningIndex] = min(distances);
        [winningRow, winningColumn, ~] = ind2sub(outputSize, winningIndex);
    
        for i = 1:outputSize(1)
            for j = 1:outputSize(2)
                ri0 = [winningRow, winningColumn];
                ri = [i, j];
                delta_r = abs(ri - ri0);
        
                % Neighbourhood function h(i,i0)
                h = exp(-sum((delta_r).^2) / (2*sigma^2));
                
                % delta_w
                w_i = squeeze(weights(i,j,:));
                delta_w = eta * h * (x' - w_i);
                
                % Update weights
                weights(i,j,:) = w_i + delta_w;
            end
        end
    end
end

% Plot
figure;
colors = 'rgb';

subplot(1, 2, 1);
for p = 1:numberOfPatterns
    x = data(p, :);
    dists = sum((reshape(initialWeights, [], numberOfFeatures) - x).^2, 2);
    [~, winningIndex] = min(dists);
    [winningRow, winningColumn, ~] = ind2sub(outputSize, winningIndex);
    scatter(winningColumn, winningRow, [], colors(labels(p)+1), 'filled');
    hold on;
end
title('Initial Random Weights', 'FontSize', 14);

subplot(1, 2, 2);
for p = 1:numberOfPatterns
    x = data(p, :);
    dists = sum((reshape(weights, [], numberOfFeatures) - x).^2, 2);
    [~, winningIndex] = min(dists);
    [winningRow, winningColumn, ~] = ind2sub(outputSize, winningIndex);
    scatter(winningColumn, winningRow, [], colors(labels(p)+1), 'filled');
    hold on;
end
title('Final Weights', 'FontSize', 14);

h1 = scatter(NaN, NaN, 'r', 'filled');
h2 = scatter(NaN, NaN, 'g', 'filled');
h3 = scatter(NaN, NaN, 'b', 'filled');
legend([h1, h2, h3], {'Setosa', 'Versicolor', 'Virginica'}, 'FontSize', 14);