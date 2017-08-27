%% Load in the data
n = 1;
a = [];
for n = 1:81
    try
        b = csvread(strcat('C:\Users\John\Desktop\Thesis Project\laughter-detector\dataset\', num2str(n), '_dataset.csv'));
        a = [a; b];
    end
end

%%
c = a(1:700:length(a),1:20);
d = a(1:700:length(a),21);
e = a(1:700:length(a),22);

%% Implimenting t-SNE

figure(1);
tsne(c, e, 2, 20, 30);
figure(2);
tsne(c, d, 2, 20, 30);
