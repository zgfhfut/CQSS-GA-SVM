%%
% *BOLD TEXT* clc;
clear all;
clc;
clear all;
addpath('voicebox');
addpath('voicebox');
% add required libraries to the path
addpath(genpath('utility'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath('voicebox');
dirName=('D:\����ASVʵ�� - ����\Baseline����\ASVspoof2017_train_dev\wav\train');
files=getAllFiles(dirName);
data=[];
for i = 1:numel(files)
    disp(i/numel(files)*100);
    filename = char(files(i));
    [x, fs] = audioread(filename);
%    

        mfc =melcepst_no_DCT(x, fs, '', nCeps, nChan, fSize, fRate);
%            mfc=cqcc(x, fs, 96, fs/2, fs/2^10, 16, 19, 'sdD')';           
[mfcsize,feature]=size(mfc);
center_frame=round(1/2*mfcsize);
liftframe=center_frame-39; 
rightframe=center_frame+40;

  mfccnew=mfc(liftframe:rightframe,:)'; %57ά*41֡
  mfccnew2=mfccnew(:) ;%ƴ��һ��  ��֡ƴ�ӣ�ǰ40��Ϊ��һ֡��40ά��Ȼ���ǵڶ�֡��40ά
mfccend=mfccnew2';
data=[data;mfccend];
end
save newCQCCtrain5780.txt data -ascii


dirName=('D:\����ASVʵ�� - ����\Baseline����\ASVspoof2017_train_dev\wav\eval');
files=getAllFiles(dirName);
data=[];
for i = 1:numel(files)
    disp(i/numel(files)*100);
    filename = char(files(i));
    [x, fs] = audioread(filename);
%    

%           mfc =melcepst_no_DCT(s, fs, '', nCeps, nChan, fSize, fRate);
           mfc=cqcc(x, fs, 96, fs/2, fs/2^10, 16, 19, 'sdD')';           
[mfcsize,feature]=size(mfc);
center_frame=round(1/2*mfcsize);
liftframe=center_frame-39; 
rightframe=center_frame+40;

  mfccnew=mfc(liftframe:rightframe,:)'; %57ά*41֡
  mfccnew2=mfccnew(:) ;%ƴ��һ��  ��֡ƴ�ӣ�ǰ40��Ϊ��һ֡��40ά��Ȼ���ǵڶ�֡��40ά
mfccend=mfccnew2';
data=[data;mfccend];
end
save newCQCCeval5780.txt data -ascii




dirName=('D:\����ASVʵ�� - ����\Baseline����\ASVspoof2017_train_dev\wav\dev');
files=getAllFiles(dirName);
data=[];
for i = 1:numel(files)
    disp(i/numel(files)*100);
    filename = char(files(i));
    [x, fs] = audioread(filename);
%    

%           mfc =melcepst_no_DCT(s, fs, '', nCeps, nChan, fSize, fRate);
           mfc=cqcc(x, fs, 96, fs/2, fs/2^10, 16, 19, 'sdD')';           
[mfcsize,feature]=size(mfc);
center_frame=round(1/2*mfcsize);
liftframe=center_frame-39; 
rightframe=center_frame+40;

  mfccnew=mfc(liftframe:rightframe,:)'; %57ά*80֡
  mfccnew2=mfccnew(:) ;%ƴ��һ��  ��֡ƴ�ӣ�ǰ40��Ϊ��һ֡��40ά��Ȼ���ǵڶ�֡��40ά
mfccend=mfccnew2';
data=[data;mfccend];
end
save newCQCCdev5780.txt data -ascii


