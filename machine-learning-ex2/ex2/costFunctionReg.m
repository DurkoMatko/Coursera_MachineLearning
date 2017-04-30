function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%hypothesis
h = sigmoid(X*theta);

%cost_func without regularization
J_withoutReg = (1/m) * (-y'*log(h) - (1-y)'*log(1-h));

J_regularization = (lambda/(2*m)) * sum(theta(2:end).^2);

%%Cost when we add regularization
J = J_withoutReg + J_regularization;


grad_withoutReg = (1/m) * (X' * (h - y));

grad_regularization = (lambda/m) .* theta(2:end);
grad_regularization = [0;grad_regularization];

grad = grad_withoutReg + grad_regularization;


%h = sigmoid(X*theta);
%shiftedTheta = theta(2:end);
%thetaReg=[0;shiftedTheta];
%J = (1/m) * (-y'*log(h) - (1-y)'*log(1-h)) + (lambda/(2*m))*thetaReg'*thetaReg;
%grad = (1/m) * (X' * (h - y) + lambda*thetaReg);
% =============================================================

end
