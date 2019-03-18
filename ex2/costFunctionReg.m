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

%Calculating Theta'X
thetaX = X*theta;
%Calculating the hx(theta)
hThetaX = sigmoid(thetaX);
%Calculating the first and second parts of the summation
first_part = -y.*log(hThetaX);
second_part= (1.-y).*log(1.-hThetaX);
J = sum(first_part-second_part)/m;

% Adding the regularization factor
reg_factor = (lambda/(2*m))*sum((theta.^2));
J= J+reg_factor-lambda/(2*m)*theta(1)^2;
%J(1) = J(1) - reg_factor(1);

grad = (hThetaX-y)'*X/m;
reg_factor_grad = (lambda/m)*theta';
grad = grad + reg_factor_grad;
grad(1) = grad(1)- reg_factor_grad(1);

grad = grad';




% =============================================================

end
