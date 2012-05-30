%% Run step
inputImage = facefactor.fetchInputImage(cam);
bbox = step(faceDetector, inputImage);
if ~isempty(bbox)
    faceBox = bbox(1, :);
    % Adjust for an apparent training error in the faceDetector.
    faceBox(3:4) = faceBox(3:4) * 1.015;
    faceImage = imcrop(inputImage, faceBox);
    faceImage = imresize(faceImage, [200 NaN]);
    faceImage = imcrop(faceImage, [20 0 159 200]);
    eyesImage = imcrop(faceImage, [20 58 119 39]);
    faceImage = imadjust(faceImage, stretchlim(eyesImage, [0.001 0.999]));
    faceImage = immultiply(im2double(faceImage), mask);
else
    faceImage = mask;
end

eyesImage = imadjust(eyesImage);
eyesImage = im2double(eyesImage);
eyesBlank = mean(mean(eyesImage));
eyesImage = eyesImage - eyesBlank;
eyesImage = eyesImage .* eyesMask;
eyesImage = eyesImage + eyesBlank;
%% Detect MSER features
regions = detectMSERFeatures(eyesImage, 'RegionAreaRange', [35 530], 'MaxAreaVariation', 0.2);
levels = arrayfun(@(x, y) eyesImage(floor(y), floor(x)), regions.Centroid(:, 1), regions.Centroid(:, 2));
regions = regions(levels < eyesBlank);

subplot(2, 3, [1 2 4 5]); subimage(inputImage); axis off;
subplot(2, 3, 3); subimage(eyesImage); hold on;
plot(regions);
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
