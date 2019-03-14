function plot_pca_filters(U)

global patch_size;

% plot filters in a big image
big_filters = min(min(U))*ones((patch_size+1)*patch_size-1);
for i=1:patch_size,
  for j=1:patch_size,
    big_filters(((i-1)*(patch_size+1)+1):(i*(patch_size+1)-1),...
      ((j-1)*(patch_size+1)+1):(j*(patch_size+1)-1)) = ...
      reshape(U(:,(i-1)*patch_size+j), patch_size, patch_size);
  end
end
imagesc(big_filters); colormap(gray);
axis square;
axis off;