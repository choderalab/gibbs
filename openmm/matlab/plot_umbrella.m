% Display distribution of samples from each umbrella state, coloring each a different color.

clear;
clf ; hold on;
nstates = 100; % number of umbrellas
X = textread('output/2d-repex-umbrella-3sigma.out', '%f', 'headerlines', 11);
Y = reshape(X, 200, size(X,1)/200)';
Y = Y(1:end,:);
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
%colors = hsv(nstates);
for i = 1:nstates
  plot(Y(:,2*i-1), Y(:,2*i), sprintf('%c.', colors(mod(i,7)+1)));  
  %plot(Y(:,2*i-1), Y(:,2*i), '.', 'markeredgecolor', colors(i,:), 'markerfacecolor', colors(i,:));  
end
axis square; axis([-180 +180 -180 +180])
%colorbar;


