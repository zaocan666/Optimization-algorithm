x=-10:0.1:10;
y=x;
[X,Y]=meshgrid(x,y);
f=-Y.*sin(X)-X.*cos(Y);
surf(X,Y,f);