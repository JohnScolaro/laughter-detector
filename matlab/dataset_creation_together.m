laughter_data = [];
laughter_labels = [];
clip_labels = [];
laughter_labels_nn = [];

% The number of audio files we are processing
files = 81;
% A list of files which MFCC's dont work for
bad_files = [];
% A list of the lengths of the data
data_length = [];

for number = 1:files
    
    num = num2str(number);
    disp(num);
    disp('Creating Data File');
    
    % Load song
    [y, Fs] = audioread(strcat('C:\Users\John\Desktop\Thesis Project\laughter-detector\modified_audio\', num, '.wav'));

    % MFCCs
    [cepstra,aspectrum,pspectrum] = melfcc(y, Fs, 'maxfreq', 8000, 'numcep', 20, 'nbands', 22, 'wintime', 0.02, 'hoptime', 0.02);
    data_length = [data_length, length(cepstra)];
    
    % For the cepstra matrix, take each row and subtract the mean, and
    % divide by the standard deviation to normalize it.
    for i = 1:20
        cepstra(i,:) = (cepstra(i,:) - mean(cepstra(i,:))) / std(cepstra(i,:));
    end
    
    % Calculate deltas
    % del = deltas(cepstra)/60;
    % Double deltas are deltas applied twice with a shorter window
    % ddel = deltas(deltas(cepstra,5)/10,5)/10;
    

    % Test to see if melfcc has returned NaN. If so, add to bad_files list and skip.
    [testx, testy] = find(isnan(cepstra));
    if (length(testx) > 10)
        bad_files = [bad_files, number];
        disp('This is a bad file');
        continue
    end
    
    % Join features together
    % cept_d_dd = [cepstra;del;ddel];
    cept_d_dd = cepstra;
    
    disp('Creating Clip Label File');

    recording_labels = ones(1, data_length(number)) * number;

    disp('Creating Label File for Neural Networks');

    % Load laughter_times. Values in ms.
    try
        laughter_times = csvread(strcat('../modified_audio/john_scolaro/', num, '.ltimes'));
    catch
        disp('Could not find laughter times file, or file is empty.');
        laughter_times = [];
    end
    laughter_times = transpose(round(laughter_times / 20));

    % Find labels
    [a, b] = size(laughter_times);
    % Set all to [1; 0]
    nn_labels = [ones(1, data_length(number)); zeros(1, data_length(number))];
    for x = 1:b
        for z = laughter_times(1, x):laughter_times(2, x)
            % Set specifically to [0; 1] when laughter
            nn_labels(:,z) = [0;1];
        end
    end
    
    % Create last vector, counting upwards. This is used for timing.
    count = 1:1:length(cept_d_dd);
    
    csvwrite(strcat(num, '_dataset.csv'), [transpose(cept_d_dd), transpose(recording_labels), transpose(count), transpose(nn_labels)]);

end


