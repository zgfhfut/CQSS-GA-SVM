clc; clear all; close all;

%% ADD CQT TOOLBOX TO THE PATH
addpath('CQT_toolbox_2013');

%device_model = {'HTC D610t','HTC D820t','HTC M7','Huawei Honor6','Huawei Honor7',...
    %'Huawei Mate7','iPhone 4s','iPhone 5','iPhone 5s','iPhone 6','iPhone 6s',...
   % 'Meizu ML note','Meizu MX2','Meizu MX4','Mi 3','Mi 4','Mi HM note1','Mi HM note2',...
   % 'OPPO Find7','OPPO Oneplus','OPPO R831S','Samsung Galaxy Note2','Samsung Galaxy S5',...
    %'Samsung GT-I8558'};
device_model={'pure_audio_true1','pure_audio_false1'}

% device_model = {'iPhone 5s','iPhone 6','iPhone 6s',...
%     'Meizu ML note','Meizu MX2','Meizu MX4','Mi 3','Mi 4','Mi HM note1','Mi HM note2',...
%     'OPPO Find7','OPPO Oneplus','OPPO R831S','Samsung Galaxy Note2','Samsung Galaxy S5',...
%     'Samsung GT-I8558'};
    

    CQT_mean=[];
    CQT_total=[];
    data = [];
for device_num = 1:length(device_model)
%  for device_num = 1:1
    
    tic;

    device_label = device_model{device_num};
   % file_dir_near_silence = ['E:\data\yinpincopy'];
    file_dir_near_silence = ['E:\data\' device_label];
% file_dir_near_silence = ['D:\手机录音设备识别数据库\Short Sample Database - 3s-including noise\CKC Database\Test\clean\' device_label];

    files_near_silence = getAllFiles(file_dir_near_silence);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    
    sample_num = 100;
                                                                                                                                                                                                                                                                                                                  
    for j = 1:sample_num

        disp(j/numel(files_near_silence));
        filename_near_silence = char(files_near_silence(j));

        [s,fs] = audioread(filename_near_silence);
        s = s(:,1);
        
        
%         %预加重
         premcoef = 0.97;
%             %s = rm_dc_n_dither(s, fs); 
         s = filter([1 -premcoef], 1, s);

%% PARAMETERS
B = 60;
fmax = fs/2;
fmin = fmax/2^3;% fmin = fmax/2^7;
d = 16;
cf = 19;
ZsdD = 'sd';

% COMPUTE CQCC FEATURES
        %[CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec] = ...
        %cqcc(s, fs, B, fmax, fmin, d, cf, ZsdD);  

        %%CQT尺寸为1*n
        [LogP_absCQT] = cqcc(s, fs, B, fmax, fmin);  
        CQT_mean(j,:)=mean(LogP_absCQT');
        
%         %%CQT尺寸为m*n
%         [LogP_absCQT] = cqcc(s, fs, B, fmax, fmin);
%         LogPabsCQT = LogP_absCQT';
%         [framenum,featurenum]=size(LogPabsCQT);
%         center_frame=round(1/2*framenum);
%         liftframe=center_frame-9; 
%         rightframe=center_frame+10;
%         LogPabsCQTnew=LogPabsCQT(liftframe:rightframe,:)'; %419维*20帧
%         LogPabsCQTnew2=LogPabsCQTnew(:) ;%拼成一列  安帧拼接，前419个为第一帧的419维，然后是第二帧的419维
%         LogPabsCQTend=LogPabsCQTnew2';
%         CQT_mean(j,:)=[0,LogPabsCQTend];
       
       %%
        clear s_angle
        clear CQcc 
        
    end
      
      CQT_total = [CQT_total;CQT_mean];                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
 
      
      toc;
end

% save CQT480D20frame_test3s200_clean.txt CQT_total -ascii
save CQT180D_train3s2000.txt CQT_total -ascii
% csvwrite('CQCC19X10D_NORES_BT_d_CKC150.csv',CQCC_total);








   