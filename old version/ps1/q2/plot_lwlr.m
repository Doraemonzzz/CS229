function plot_lwlr(X, y, tau, res)


x = zeros(2,1);
for i=1:res,
  for j=1:res,
    x(1) = 2*(i-1)/(res-1) - 1;
    x(2) = 2*(j-1)/(res-1) - 1;
    pred(j,i) = lwlr(X, y, x, tau);
  end
end

figure(1);
clf;
axis off;
hold on;
imagesc(pred, [-0.4 1.3]);
plot((res/2)*(1+X(y==0,1))+0.5, (res/2)*(1+X(y==0,2))+0.5, 'ko');
plot((res/2)*(1+X(y==1,1))+0.5, (res/2)*(1+X(y==1,2))+0.5, 'kx');
axis equal;
axis square;
text(res/2 - res/7, res + res/20, ['tau = ' num2str(tau)], 'FontSize', 18);


