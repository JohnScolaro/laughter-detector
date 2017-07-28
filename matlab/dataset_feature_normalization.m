%% Init

% The number of audio files we are processing
files = 81;
% A list of files which MFCC's dont work for
bad_files = [];
% A list of the lengths of the data
data_length = [];

%% One hist with all the features

for number = 1:files
    
    num = num2str(number);
    disp(num);
    disp('Creating Data File');
    
    % Load song
    [y, Fs] = audioread(strcat(num, '.wav'));
    
    % MFCCs
    [cepstra,aspectrum,pspectrum] = melfcc(y, Fs, 'maxfreq', 8000, 'numcep', 20, 'nbands', 22, 'wintime', 0.02, 'hoptime', 0.02);
    
    [testx, testy] = find(isnan(cepstra));
    if (length(testx) > 10)
        bad_files = [bad_files, number];
        disp('This is a bad file');
        continue
    end
    
    figure('rend','painters','pos',[10 10 900 600])
    hist(transpose(cepstra))
    legend('Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7', 'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13', 'Feature 14', 'Feature 15', 'Feature 16', 'Feature 17', 'Feature 18', 'Feature 19', 'Feature 20');
    print(num2str(number), '-dpng')
end

%% Many hists. One for each feature.

for number = 1:files
    num = num2str(number);
    disp(num);
    
    % Load song
    [y, Fs] = audioread(strcat(num, '.wav'));
    
    % MFCCs
    [cepstra,aspectrum,pspectrum] = melfcc(y, Fs, 'maxfreq', 8000, 'numcep', 20, 'nbands', 22, 'wintime', 0.02, 'hoptime', 0.02);
    figure('rend','painters','pos',[10 10 1200 800])

    for col = 1:20
        subplot(4, 5, col);
        leg = strcat('Feature ', num2str(col));
        title(leg)
        hist(transpose(cepstra(col,:)))
        axis([-50, 50, 0, 80000])
        axis 'auto y'
    end

    print(num2str(number), '-dpng')
end

