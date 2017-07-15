load('laughter_data');
load('laughter_labels');
load('clip_labels');

%%

laughter_data = laughter_data(1:500000,1:end);
laughter_labels = laughter_labels(1:500000,1:end);
clip_labels = clip_labels(1:500000,1:end);