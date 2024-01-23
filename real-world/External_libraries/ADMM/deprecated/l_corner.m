function reg_c = l_corner(rho,eta,reg_param)
%L_CORNER Locate the "corner" of the L-curve.
%
% reg_c = l_corner(rho,eta,reg_param)
%
% Locates the "corner" of the L-curve in log-log scale.
%
% It is assumed that corresponding values of || A x - b ||, || L x ||,
% and the regularization parameter are stored in the arrays rho, eta,
% and reg_param, respectively (such as the output from routine l_curve).
%
% Per Christian Hansen, DTU Compute, January 31, 2015.

  % Compute the curvature of L-curve.
    dlogrho  = gradient(log(rho));
    ddlogrho = gradient(dlogrho);
    dlogeta  = gradient(log(eta));
    ddlogeta = gradient(dlogeta);
    num   = dlogrho .* ddlogeta - ddlogrho .* dlogeta;
    denom = (dlogrho.^2 + dlogeta.^2).^(1.5);
    kappa = - 2*num ./ denom;
    kappa(denom < 0) = NaN;

    % Locate the corner.  If the curvature is negative everywhere,
    % then define the leftmost point of the L-curve as the corner.
    [kappa_max,ki] = max(kappa);
    if (kappa_max < 0)
      reg_c = reg_param(end);
    else
      reg_c = reg_param(ki);
    end
end