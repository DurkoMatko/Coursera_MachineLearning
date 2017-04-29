function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp_X = X(:,2);
for iter2 = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    partialTheta1=0;
    partialTheta2=0;
    
    for iter = 1: m
      partialTheta1 = partialTheta1 + ((theta(1) + theta(2) * temp_X(iter)) - y(iter));
      partialTheta2 = partialTheta2 + ((theta(1) + theta(2) * temp_X(iter)) - y(iter))*temp_X(iter);
    endfor
    
    %fprintf("partialTheta1\n");
    %partialTheta1
    %fprintf("partialTheta2\n");
    %partialTheta2

    theta(1) -= (alpha/m)*partialTheta1;
    theta(2) -= (alpha/m)*partialTheta2;

    %pause;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter2) = computeCost(X, y, theta);

end

end
