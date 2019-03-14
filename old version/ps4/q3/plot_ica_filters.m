function plot_ica_filters(W)

global patch_size;
global W_z;

% sort filters by 2-norm
F = W*W_z;
for i=1:size(W),
  norms(i) = norm(F(i,:));
end
[norms,idxs] = sort(norms, 'ascend');

% plot filters in a big image
big_filters = min(min(W))*ones((patch_size+1)*patch_size-1);
for i=1:patch_size,
  for j=1:patch_size,
    big_filters(((i-1)*(patch_size+1)+1):(i*(patch_size+1)-1),...
      ((j-1)*(patch_size+1)+1):(j*(patch_size+1)-1)) = ...
      reshape(W(idxs((i-1)*patch_size+j),:), patch_size, patch_size);
  end
end
imagesc(big_filters); colormap(gray);
axis square;
axis off;