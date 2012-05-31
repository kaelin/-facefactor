%% Run step
inputImage = facefactor.fetchInputImage(cam);

%% Preprocess image
% tic;
bbox = faceDetector.step(inputImage);
if ~isempty(bbox)
    faceBox = bbox(1, :);
    % Adjust for an apparent training error in the faceDetector.
    faceBox(3:4) = faceBox(3:4) * 1.015;
    faceImage = imcrop(inputImage, faceBox);
    faceImage = imresize(faceImage, [200 NaN]);
    faceImage = imcrop(faceImage, [20 0 159 200]);
    eyesImage = imcrop(faceImage, eyesCrop);
    faceImage = imadjust(faceImage, stretchlim(eyesImage, [0.001 0.999]));
    faceImage = immultiply(im2double(faceImage), faceMask);
else
    faceImage = faceMask;
end

eyesImage = im2double(eyesImage);
eyesBlank = mean(mean(eyesImage));
eyesImage = eyesImage - eyesBlank;
eyesImage = eyesImage .* eyesMask;
eyesImage = eyesImage + 0.999; % eyesBlank;
eyesImage = imadjust(eyesImage);

%% Detect MSER features
regionsB4 = detectMSERFeatures(eyesImage, 'RegionAreaRange', [20 350], 'MaxAreaVariation', 0.25);

%% Postprocess and cluster MSER features
regions = regionsB4;
levels = facefactor.sampleMSERRegions(eyesImage, regions);
regions = facefactor.selectMSERRegions(regions, levels < 0.7);
sides = arrayfun(@(x) sign(x - 60), regions.Centroid(:, 1));
regionsLt = facefactor.selectMSERRegions(regions, sides == -1);
regionsRt = facefactor.selectMSERRegions(regions, sides == 1);
[CLt, MLt] = facefactor.clusterMSERRegions(regionsLt, 25, 3);
[CRt, MRt] = facefactor.clusterMSERRegions(regionsRt, 25, 3);

%% Visualize results
% toc;
subplot(2, 3, [1 2 4 5]); subimage(inputImage); axis off;
subplot(2, 3, 3); subimage(imcrop(faceImage, eyesCrop) + 0.7); hold all;
% plot(regionsLt); hold all;
% plot(regionsRt); hold all;
if ~isempty(CLt)
    scatter(CLt(:, 1), CLt(:, 2), (MLt * 3).^1.7, 'Filled'); hold all;
end
if ~isempty(CRt)
    scatter(CRt(:, 1), CRt(:, 2), (MRt * 3).^1.7, 'Filled'); hold all;
end
hold off;
subplot(2, 3, 6); subimage(faceImage); axis off;
if rec.Training.isvalid
    [label, score] = rec.step(faceImage);
    if label ~= 0
        title(rec.LabelNames{label});
    else
        title('?');
    end
end
drawnow;
