function y = thresh_markus(x,sorh,t)
% Perform soft hard or soft positive thresholding. 
% changed Markus Haltmeier, Sept 2007

switch sorh
  case 's'
    tmp = (abs(x)-t);
    tmp = (tmp+abs(tmp))/2;
    y   = sign(x).*tmp;
    
    case 'p'
    tmp = (abs(x)-t);
    y = max(0,tmp);

    case 'sp'
    tmp = (abs(x)-t);
    tmp = (tmp+abs(tmp))/2;
    y   = sign(x).*tmp;
    y( x < 0 ) = 0;
 
  case 'h'
    y   = x.*(abs(x)>t);
 
  otherwise
    error('Invalid argument value.')
end
