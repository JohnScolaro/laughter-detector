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

disp('Creating Data File');
for number = 1:files
    
    num = num2str(number);

    % Load song
    [y, Fs] = audioread(strcat(num, '.wav'));

    % MFCCs
    [cepstra,aspectrum,pspectrum] = melfcc(y, Fs, 'maxfreq', 8000, 'numcep', 20, 'nbands', 22, 'wintime', 0.02, 'hoptime', 0.02);
    data_length = [data_length, length(cepstra)];
    
    % Calculate deltas
    del = deltas(cepstra);
    % Double deltas are deltas applied twice with a shorter window
    ddel = deltas(deltas(cepstra,5),5);

    % Test to see if melfcc has returned NaN. If so, add to bad_files list and skip.
    [testx, testy] = find(isnan(cepstra));
    if (length(testx) > 10)
        bad_files = [bad_files, number];
        disp('This is a bad file');
        continue
    end
    
    % Join features together
    cept_d_dd = [cepstra;del;ddel];
    
    laughter_data = [laughter_data, cept_d_dd];
    
    csvwrite(strcat(num, '_data.csv'), transpose(cept_d_dd));
    disp(num);
end

% Transpose our file, write it to disk, and clear the memory.
laughter_data = transpose(laughter_data);
csvwrite('laughter_data.csv', laughter_data);
clear('laughter_data');
clear('cept_d_dd');

disp('Creating Label File');
for number = 1:files
    
    num = num2str(number);
    
    % Check if this audio file is a bad one. If so, skip it.
    flag = 0;
    [a, b] = size(bad_files);
    for x = 1:b
        if (number == bad_files(x))
            flag = 1;
        end
    end
    if (flag == 1)
        disp('Bad File');
        continue
    end
    
    % Load laughter_times. Values in ms.
    try
        laughter_times = csvread(strcat(num, '.ltimes'));
    catch
        laughter_times = [];
    end
    laughter_times = transpose(round(laughter_times / 20));

    % Find labels
    [a, b] = size(laughter_times);
    labels = zeros(1, data_length(number));
    for x = 1:b
        for z = laughter_times(1, x):laughter_times(2, x)
            labels(z) = 1;
        end
    end
    
    laughter_labels = [laughter_labels, labels];
    csvwrite(strcat(num, '_labels.csv'), transpose(labels));
    
    disp(num);
end

% Write the labels to disk and clear the memory.
laughter_labels = transpose(laughter_labels);
csvwrite('laughter_labels.csv', laughter_labels);
clear('laughter_labels');
clear('labels');

disp('Creating Clip Label File');
for number = 1:files
    
    num = num2str(number);
    
    %Check if this audio file is a bad one. If so, skip it.
    flag = 0;
    [a, b] = size(bad_files);
    for x = 1:b
        if (number == bad_files(x))
            flag = 1;
        end
    end
    if (flag == 1)
        disp('Bad File');
        continue
    end

    recording_labels = ones(1,data_length(number)) * number;
    
    clip_labels = [clip_labels, recording_labels];
    csvwrite(strcat(num, '_clip_labels.csv'), transpose(recording_labels));
    
    disp(num);
end

% Write the labels to disk and clear the memory.
clip_labels = transpose(clip_labels);
csvwrite('clip_labels.csv', clip_labels);
clear('clip_labels');
clear('recording_labels');

disp('Creating Label File for Neural Networks');
for number = 1:files
    
    num = num2str(number);
    
    %Check if this audio file is a bad one. If so, skip it.
    flag = 0;
    [a, b] = size(bad_files);
    for x = 1:b
        if (number == bad_files(x))
            flag = 1;
        end
    end
    if (flag == 1)
        disp('Bad File');
        continue
    end
    
    % Load laughter_times. Values in ms.
    try
        laughter_times = csvread(strcat(num, '.ltimes'));
    catch
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
    
    laughter_labels_nn = [laughter_labels_nn, nn_labels];
    csvwrite(strcat(num, '_nn_labels.csv'), transpose(nn_labels));
    
    disp(num);
end

% Write the labels to disk and clear the memory.
laughter_labels_nn = transpose(laughter_labels_nn);
csvwrite('laughter_labels_nn.csv', laughter_labels_nn);
clear('laughter_labels_nn');
clear('nn_labels');


