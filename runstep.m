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

%% Infer image rotation
figure(2); clf;
inferImage = inputImage;
[inferAngle, inferConfidence] = facefactor.inferImageRotation(eyesImage, eyesMask, engine, gca);
if inferAngle ~= 0
    inferImage = imrotate(inferImage, double(inferAngle));
    bbox = faceDetector.step(inferImage);
    if ~isempty(bbox)
        faceBox = bbox(1, :);
        % Adjust for an apparent training error in the faceDetector.
        faceBox(3:4) = faceBox(3:4) * 1.015;
        faceImage = imcrop(inferImage, faceBox);
        faceImage = imresize(faceImage, [200 NaN]);
        faceImage = imcrop(faceImage, [20 0 159 200]);
        eyesImage = imcrop(faceImage, eyesCrop);
        faceImage = imadjust(faceImage, stretchlim(eyesImage, [0.001 0.999]));
        faceImage = immultiply(im2double(faceImage), faceMask);
    end
end
disp([inferAngle inferConfidence]);

%% Visualize results
% toc;
figure(1);
subplot(2, 3, [1 2 4 5]); subimage(inputImage); axis off;
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
