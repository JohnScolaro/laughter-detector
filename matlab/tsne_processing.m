load('laughter_data');
load('laughter_labels');
load('clip_labels');

%% Implimenting t-SNE

[a, b] = size(laughter_data);
idx = unidrnd(a, 7000, 1);
x = laughter_data(idx, :);
labels = clip_labels(idx);

%clear('laughter_data');
%clear('laughter_labels');
%clear('laughter_labels');

tsne(x, labels, 2, 60, 30);