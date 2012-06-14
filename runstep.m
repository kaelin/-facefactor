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
else
    faceImage = faceMask;
    return;
end

%% Infer image rotation
inferImage = inputImage;
[~, confidence, angle] = eyesDetector.step(faceImage);
figure(2); clf; eyesDetector.plot();
if angle ~= 0
    inferImage = imrotate(inferImage, double(angle));
    bbox = faceDetector.step(inferImage);
    if ~isempty(bbox)
        faceBox = bbox(1, :);
        % Adjust for an apparent training error in the faceDetector.
        faceBox(3:4) = faceBox(3:4) * 1.015;
        faceImage = imcrop(inferImage, faceBox);
        faceImage = imresize(faceImage, [200 NaN]);
        faceImage = imcrop(faceImage, [20 0 159 200]);
    end
end
faceImage = immultiply(im2single(faceImage), faceMask);
faceImage = imadjust(faceImage, stretchlim(faceImage, [0.001 0.999]));

% Visualize results
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
