close all; close; clc;


%% Загружем данные для обучения
elefanteFolder = 'none';%слон
fartallaFolder = 'none';%бабочка
gattoFolder = 'none';%кот
scoiattololeFolder = 'none';%белка

elefanteImages = imageDatastore(elefanteFolder, 'LabelSource', 'foldernames');
fartallaImages = imageDatastore(fartallaFolder, 'LabelSource', 'foldernames');
gattoImages = imageDatastore(gattoFolder, 'LabelSource', 'foldernames');
scoiattololeImages = imageDatastore(scoiattololeFolder, 'LabelSource', 'foldernames');

imds = imageDatastore(cat(1, elefanteImages.Files, ...
    fartallaImages.Files, gattoImages.Files, ...
    scoiattololeImages.Files), 'LabelSource', 'foldernames');

%% Выведем несколько примеров изображений
figure
numImages = 7000;
perm = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    drawnow;
end

%% Разделяем данные на наборы для обучения
[trainImages, testImages] = splitEachLabel(imds, 0.7, 'randomized');

resizedTrainImages = augmentedImageDatastore([28 28 3], trainImages, 'ColorPreprocessing', 'gray2rgb');
resizedTestImages = augmentedImageDatastore([28 28 3], testImages, 'ColorPreprocessing', 'gray2rgb');
%% Параметры обучения

options = trainingOptions('sgdm','InitialLearnRate',0.01, ...
   'MaxEpochs',8,'Shuffle','every-epoch', 'ValidationData', ...
   resizedTestImages,'ValidationFrequency',10,'Verbose', ...
   false,'Plots','training-progress');

%% Ахритектура нейросети

 layers = [
     imageInputLayer([28 28 3],'AverageImage',ones([28 28 3]));
 
     convolution2dLayer(3,25,'Padding','same')
     batchNormalizationLayer
     reluLayer
 
     maxPooling2dLayer(2,'Stride',2) % 5-й слой
 
     convolution2dLayer(2,8,'Padding','same')
     batchNormalizationLayer
     reluLayer
     
     maxPooling2dLayer(2,'Stride',2) % 9-й слой
 
     convolution2dLayer(3,15,'Padding','same')
     batchNormalizationLayer
     reluLayer
 
     fullyConnectedLayer(4)
     softmaxLayer
     classificationLayer
     ];

%% Обучение нейросети
net = trainNetwork(resizedTrainImages, layers, options);