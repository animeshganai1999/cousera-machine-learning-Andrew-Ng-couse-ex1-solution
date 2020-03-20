function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    hypothesis= (X)*theta;
    val = hypothesis - y;
    
    theta = theta - ((alpha/m).*(sum(val.*X)))';

    %theta(1) = theta(1) - (alpha/m)*sum(val);
    %theta(2) = theta(2) - (alpha/m)*sum(val.*X(:,2));
    %theta(3) = theta(3) - (alpha/m)*sum(val.*X(:,3));
      
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
