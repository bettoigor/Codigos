phi = 2.8021760794565616
dphi = 47.91729630391639
control = 10.979324075979097
  
  

x1= 0-phi;
x2 = -dphi;

alpha = 4.5;
rho = 1;
K2 = 0.25;
K2_s = 0.1;

s = x2 - alpha*x1
s_s = x2 + alpha*x1

u = -K2*x2 + rho*(sign(s))

u_s = -K2_s*x2 + rho*(sign(s_s))*(sqrt(abs(s_s)))


x = -5:0.5:5;
[X,Y] = meshgrid(x);
f = -X.^4 +4*(X.^2 - Y.^2) - 3;
surf(X,Y,f)
f = X.*exp(-X.^2 - Y.^2);
surf(X,Y,f)
hold on
contour(X,Y,f)
surf(X,Y,f)
[dx,dy] = gradient(f,0.5);

hold on
quiver(X,Y,dx,dy)

clear all
syms x y dx dy
f = -x^4 +4*(x^2 - y^2) - 3;
[dx,dy] = gradient(f,0.5);

f = X*Y;
hold off
surf(X,Y,f)
[dx,dy] = gradient(f,0.5);
quiver(X,Y,dx,dy)
hold on




