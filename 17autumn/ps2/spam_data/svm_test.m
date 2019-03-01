%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% svm_test.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct test and train matrices
Xtest = 1.0 * (Xtest > 0);
squared_X_test = sum(Xtest.^2, 2);
m_test = size(Xtest, 1);
gram_test = Xtest * Xtrain';
Ktest = full(exp(-(repmat(squared_X_test, 1, m_train) ...
                   + repmat(squared_X_train', m_test, 1) ...
                   - 2 * gram_test) / (2 * tau^2)));

% preds = Ktest * alpha;

% fprintf(1, 'Test error rate for final alpha: %1.4f\n', ...
%         sum(preds .* ytest <= 0) / length(ytest));

preds = Ktest * avg_alpha;
fprintf(1, 'Test error rate for average alpha: %1.4f\n', ...
        sum(preds .* ytest <= 0) / length(ytest));
test_error = sum(preds .* ytest <= 0) / length(ytest);
