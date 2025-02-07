close all; close; clc;


%% Загружем данные для обучения

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');


%% Выведем несколько примеров изображений
figure
numImages = 10000;
perm = randperm(numImages,16);
for i = 1:16
    subplot(4,4,i);
    imshow(imds.Files{perm(i)});
    drawnow;
end

%% Разделяем данные на наборы для обучения
numTrainingFiles = 750;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

%% Параметры обучения
%options = trainingOptions('rmsprop','InitialLearnRate',0.01, ...
%    'MaxEpochs',8,'Shuffle','every-epoch','ValidationData', imdsValidation, ...
%    'ValidationFrequency',10,'Verbose',false,'Plots','training-progress');

options = trainingOptions('rmsprop','InitialLearnRate',0.01, ...
   'MaxEpochs',30,'Shuffle','every-epoch', ...
   'ValidationFrequency',10,'Verbose',false,'Plots','training-progress');

%% Ахритектура нашей нейронки

layers = [
    imageInputLayer([28 28 1],'AverageImage',ones([28 28 1])); %Создание входного слоя для серого изображения 28-by-28 и его нормализация

    convolution2dLayer(10,30)%слой 2D свертки для сверточных нейронных сетей
    batchNormalizationLayer %уровни пакетной нормализации.
    reluLayer %- слой ректификации.

    maxPooling2dLayer(15,'Stride',15) %Максимальный слой объединения.
    

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
    ];


%% Обучение сети
net = trainNetwork(imdsTrain,layers,options);







