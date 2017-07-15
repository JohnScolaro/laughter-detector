[y,Fs] = audioread('GCSAusE02.mp3');
laugh = csvread('test_eaf_laugh.csv');
z = y(21338*(Fs/1000):22786*(Fs/1000), 1);
a = [];

for x = 1:length(laugh)/2
    z = y(laugh((x*2-1))*(Fs/1000):laugh(x*2)*(Fs/1000), 1);
    a = [a; z];
end

player = audioplayer(a,Fs);
play(player);
audiowrite('haha.wav',transpose(a),Fs);