[y,Fs] = audioread('MEBH4M1_Sanitised.wav');
player = audioplayer(y(4*end/100:5*end/100, 1), Fs);
play(player);
spectrogram(y(4*end/100:5*end/100, 1), 1024, 800, 'yaxis');