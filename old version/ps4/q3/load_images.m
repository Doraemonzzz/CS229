function [X_ica, X_pca] = load_images

global patch_size;
global W_z;
patch_size = 16;

X_ica = zeros(patch_size*patch_size, 40000);
idx = 1;
for i=1:4,
  image = double(imread(['images/' num2str(i) '.jpg']));
  [y,x] = size(image);
  for i=1:floor(y/patch_size),
    for j=1:floor(x/patch_size),
      patch = image((i-1)*patch_size+1:i*patch_size,(j-1)*patch_size+1:j*patch_size);
      X_ica(:,idx) = reshape(patch, patch_size*patch_size, 1);
      idx = idx + 1;
    end
  end
end
X_ica = X_ica(:,1:idx-1);
W_z = ((1/size(X_ica,2))*X_ica*X_ica')^-(1/2);
X_ica = X_ica - repmat(mean(X_ica')', 1, size(X_ica,2));
X_pca = X_ica;

X_ica = 2*W_z*X_ica;
X_pca = X_pca ./ repmat(std(X_pca')', 1, size(X_pca,2));
