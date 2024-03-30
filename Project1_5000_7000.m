a     = 4;           
tb    = 0.07;        
n     = 7000; % each bit 70 samples and we have 100 bits  
ts    = 0.01;        
fs    = 1/ts;        
t_tot = 70;                      
t     = 0.01 : 0.01 : t_tot;         
f     = -fs/2 : 1/t_tot : fs/2 -1/t_tot; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = randi ([0,1], 5000, 101);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

polar_nrz=((2*data)-1)*a;
unipolar_nrz=a*data;
%Upsampling is a process of increasing the sample rate of a signal by inserting zeros between the original samples.
polar_rz= transpose(upsample(transpose(polar_nrz), 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

unipolar_nrz=repelem(unipolar_nrz, 1,70);
polar_nrz = repelem(polar_nrz, 1, 70);
polar_rz = repelem(polar_rz, 1, repmat([40 30], 1, 101));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

delay_polar_nrz = randi([0 69],5000,1);
delay_unipolar  = randi([0 69],5000,1);
delay_polar_rz  = randi([0 69],5000,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Polar_nrz_delayed = zeros(5000,7000);
unipolar_nrz_delayed = zeros(5000,7000);
Polar_rz_delayed = zeros(5000,7000);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%if delay =0 the output start from 1 to 7000 (from input 8 to 7007)
for j = 1:5000
    unipolar_nrz_delayed(j,:)= unipolar_nrz  (j, 71-delay_unipolar(j) : 7070-delay_unipolar(j));
    Polar_nrz_delayed(j,:)   = polar_nrz(j, 71-delay_polar_nrz(j) : 7070-delay_polar_nrz(j));
    Polar_rz_delayed(j,:)   = polar_rz(j, 71-delay_polar_rz(j) : 7070-delay_polar_rz(j));
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('Name','Line Codes Generation');
subplot(3,1,1);
stairs(unipolar_nrz_delayed(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('unipolar nrz');
subplot(3,1,2);
stairs(Polar_nrz_delayed(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('polar nrz');
subplot(3,1,3);
stairs(Polar_rz_delayed(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('polar rz');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Statistical Mean of Line Codes
unipolar_nrz_mean   = zeros(1,7000);     
polar_nrz_mean      = zeros(1,7000);     
polar_rz_mean       = zeros(1,7000);     
for j = 1 : 5000
unipolar_nrz_mean   = unipolar_nrz_mean + unipolar_nrz_delayed(j, :);
polar_nrz_mean      = polar_nrz_mean + Polar_nrz_delayed(j, :);
polar_rz_mean       = polar_rz_mean + Polar_rz_delayed(j, :);
end
unipolar_nrz_mean   = unipolar_nrz_mean/5000;
polar_nrz_mean      = polar_nrz_mean/5000;
polar_rz_mean       = polar_rz_mean/5000;


figure('Name','Statistical Mean of Line Codes');
subplot(3,1,1);
plot(unipolar_nrz_mean(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('Statistical mean of unipolar nrz');
subplot(3,1,2);
plot(polar_nrz_mean(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('Statistical mean of polar nrz');
subplot(3,1,3);
plot(polar_rz_mean(1, 1:n));
grid on
axis([1 n -6 6]);
xlabel('samples');
ylabel('magnitude');
title('Statistical mean of polar rz');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculating Time Mean of all waveforms
unipolar_nrz_mean_t = zeros(5000,1);     
polar_nrz_mean_t    = zeros(5000,1);     
polar_rz_mean_t     = zeros(5000,1);    
for j = 1 : 7000
unipolar_nrz_mean_t   = unipolar_nrz_mean_t + unipolar_nrz_delayed(:, j);
polar_nrz_mean_t      = polar_nrz_mean_t + Polar_nrz_delayed(:, j);
polar_rz_mean_t       = polar_rz_mean_t + Polar_rz_delayed(:, j);
end
unipolar_nrz_mean_t   = unipolar_nrz_mean_t/7000;
polar_nrz_mean_t      = polar_nrz_mean_t/7000;
polar_rz_mean_t       = polar_rz_mean_t/7000;

figure('Name','Time Mean of Line Codes');
subplot(3,1,1);
plot(unipolar_nrz_mean_t);
grid on
axis([1 5000 -6 6]);
xlabel('time in samples');
ylabel('magnitude');
title('Time mean of unipolar nrz');
subplot(3,1,2);
plot(polar_nrz_mean_t);
grid on
axis([1 5000 -6 6]);
xlabel('time in samples');
ylabel('magnitude');
title('Time mean of polar nrz');
subplot(3,1,3);
plot(polar_rz_mean_t);
grid on
axis([1 5000 -6 6]);
xlabel('time in samples');
ylabel('magnitude');
title('Time mean of polar rz');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculating statistical Autocorrelation
s_t1 = 3501;       
taw = -3500: 3499;  
R_unipolar_nrz = zeros (1, 7000);    
R_polar_nrz = zeros (1, 7000);    
R_polar_rz = zeros (1, 7000);    
for i = taw
M = i + s_t1; 
 for j = 1:5000
 R_unipolar_nrz(M) = R_unipolar_nrz(M) + unipolar_nrz_delayed(j,s_t1)* unipolar_nrz_delayed(j,s_t1+i);
 R_polar_nrz(M)    = R_polar_nrz(M)    + Polar_nrz_delayed(j,s_t1)   * Polar_nrz_delayed(j,s_t1+i);
 R_polar_rz(M)     = R_polar_rz(M)     + Polar_rz_delayed(j,s_t1)    * Polar_rz_delayed(j,s_t1+i);
 end
  R_unipolar_nrz(M) = R_unipolar_nrz(M)/5000;
  R_polar_nrz(M) = R_polar_nrz(M)/5000;
  R_polar_rz(M) = R_polar_rz(M)/5000;
end
figure('Name','Statistical Autocorrelation');
subplot(3,1,1);
plot(taw,R_unipolar_nrz);
grid on
axis([-1000 1000 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Statistical Autocorrelation of unipolar nrz');
subplot(3,1,2);
plot(taw,R_polar_nrz);
grid on
axis([-1000 1001 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Statistical Autocorrelation of polar nrz');
subplot(3,1,3);
plot(taw,R_polar_rz);
grid on
axis([-1000 1000 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Statistical Autocorrelation of polar rz');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ACF at constant tau to prove that it is stationary
R_unipolar_nrz_0 = zeros (1, 7000);    
R_polar_nrz_0 = zeros (1, 7000);    
R_polar_rz_0 = zeros (1, 7000); 
tau = 5;
for i = 1 : 7000
    sum_1_R = 0;    
    sum_2_R = 0;       
    sum_3_R = 0; 
 for j = 1 : 5000
     if (i+tau < 7001) %to avoid exceeding the matrix bounds
 sum_1_R = sum_1_R + unipolar_nrz_delayed(j,i)* unipolar_nrz_delayed(j,i+tau);
 sum_2_R = sum_2_R + Polar_nrz_delayed(j,i)* Polar_nrz_delayed(j,i+tau);
 sum_3_R = sum_3_R + Polar_rz_delayed(j,i)* Polar_rz_delayed(j,i+tau);
     end
 end 
  R_unipolar_nrz_0(i) = sum_1_R/5000;
  R_polar_nrz_0(i) = sum_2_R/5000;
  R_polar_rz_0(i) = sum_3_R/5000;
end
figure('Name','statistical ACF for constatnt taw');
subplot(3,1,1);
axis('auto');
plot(R_unipolar_nrz_0(1, 1:n-tau));
grid on
axis([1 n -20 20]);
xlabel('samples');
ylabel('magnitude');
title('statistical ACF of unipolar nrz for constatnt taw');
subplot(3,1,2);
plot(R_polar_nrz_0(1, 1:n-tau));
grid on
axis([1 n -20 20]);
xlabel('samples');
ylabel('magnitude');
title('statistical ACF of polar nrz for constatnt taw');
subplot(3,1,3);
plot(R_polar_rz_0(1, 1:n-tau));
grid on
axis([1 n -20 20]);
xlabel('samples');
ylabel('magnitude');
title('statistical ACF of polar rz for constatnt taw');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%time autocorrelation
R_unipolar_nrz_t = zeros (1, 3000);    
R_polar_nrz_t = zeros (1, 3000);    
R_polar_rz_t = zeros (1, 3000);    
taw2 = -1500: 1499;         
for i = taw2
M = i + 1501; 
 for n1 = 1501: 5500       
 R_unipolar_nrz_t(M) = R_unipolar_nrz_t(M) + unipolar_nrz_delayed(1,n1)* unipolar_nrz_delayed(1,n1+i);
 R_polar_nrz_t(M) = R_polar_nrz_t(M) + Polar_nrz_delayed(1,n1)* Polar_nrz_delayed(1,n1+i);
 R_polar_rz_t(M) = R_polar_rz_t(M) + Polar_rz_delayed(1,n1)* Polar_rz_delayed(1,n1+i);
 end
  R_unipolar_nrz_t(M) = R_unipolar_nrz_t(M)/4000;
  R_polar_nrz_t(M) = R_polar_nrz_t(M)/4000;
  R_polar_rz_t(M) = R_polar_rz_t(M)/4000;
end

figure('Name','Time Autocorrelation');
subplot(3,1,1);
plot(taw2,R_unipolar_nrz_t);
grid on
axis([-1000 1000 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Time Autocorrelation of unipolar nrz');
subplot(3,1,2);
plot(taw2,R_polar_nrz_t);
grid on
axis([-1000 1000 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Time Autocorrelation of polar nrz');
subplot(3,1,3);
plot(taw2,R_polar_rz_t);
grid on
axis([-1000 1000 -4 20]);
xlabel('time in samples');
ylabel('magnitude');
title('Time Autocorrelation of polar rz');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate the BW from PSD graph
unipolar_nrz_coe = fft(R_unipolar_nrz)/n;
polar_nrz_coe = fft(R_polar_nrz)/n;
polar_rz_coe = fft(R_polar_rz)/n;

amplitude_unipolar_nrz= sqrt(unipolar_nrz_coe.*conj(unipolar_nrz_coe));
amplitude_polar_nrz = sqrt(polar_nrz_coe.*conj(polar_nrz_coe));
amplitude_polar_rz = sqrt(polar_rz_coe.*conj(polar_rz_coe));

amp_unipolar_nrz = zeros(1,7000);
amp_polar_nrz = zeros(1,7000);
amp_polar_rz = zeros(1,7000);
% we split the matrix in halves one frome -fs/2 to 0 and the other from 0 to fs/2
for i = 1: 3501 
    amp_unipolar_nrz(i) = amplitude_unipolar_nrz(3502-i);
    amp_polar_nrz(i) = amplitude_polar_nrz(3502-i);
    amp_polar_rz(i) = amplitude_polar_rz(3502-i);
end
for i = 3502: 7000 
    amp_unipolar_nrz(i) = amplitude_unipolar_nrz(10502-i);
    amp_polar_nrz(i) = amplitude_polar_nrz(10502-i);
    amp_polar_rz(i) = amplitude_polar_rz(10502-i);
end

x =(-7000/2:(7000/2)-1) ;
x_values =(fs*x)/7000;
figure('Name','power spectral density');
subplot(3,1,1);
plot(x_values,amp_unipolar_nrz);
grid on
xlabel('freq');
ylabel('magnitude');
title('PSD of unipolar nrz');
subplot(3,1,2);
plot(x_values,amp_polar_nrz);
grid on
xlabel('freq');
ylabel('magnitude');
title('PSD of polar nrz');
subplot(3,1,3);
plot(x_values,amp_polar_rz) ;
grid on
xlabel('freq');
ylabel('magnitude');
title('PSD of polar rz');
