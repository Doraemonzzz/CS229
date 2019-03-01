%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_train.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('MATRIX.TRAIN.%d', num_train));
Xtrain = full(sparseTrainMatrix);
m_train = size(Xtrain, 1);
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (Xtrain > 0);

squared_X_train = sum(Xtrain.^2, 2);
gram_train = Xtrain * Xtrain';
tau = 8;

% Get full training matrix for kernels using vectorized code.
Ktrain = full(exp(-(repmat(squared_X_train, 1, m_train) ...
                    + repmat(squared_X_train', m_train, 1) ...
                    - 2 * gram_train) / (2 * tau^2)));

lambda = 1 / (64 * m_train);
num_outer_loops = 40;
alpha = zeros(m_train, 1);

avg_alpha = zeros(m_train, 1);
Imat = eye(m_train);

count = 0;
for ii = 1:(num_outer_loops * m_train)
  count = count + 1;
  ind = ceil(rand * m_train);
  margin = ytrain(ind) * Ktrain(ind, :) * alpha;
  g = -(margin < 1) * ytrain(ind) * Ktrain(:, ind) + ...
      m_train * lambda * (Ktrain(:, ind) * alpha(ind));
  % g(ind) = g(ind) + m_train * lambda * Ktrain(ind,:) * alpha;
  alpha = alpha - g / sqrt(count);
  avg_alpha = avg_alpha + alpha;
end
avg_alpha = avg_alpha / (num_outer_loops * m_train);
