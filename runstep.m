%% Run step
inputImage = facefactor.fetchInputImage(cam);
bbox = step(faceDetector, inputImage);
if ~isempty(bbox)
    faceImage = imcrop(inputImage, bbox(1,:));
    faceImage = imresize(faceImage, [200 NaN]);
    faceImage = imcrop(faceImage, [20 0 159 200]);
    eyesImage = imcrop(faceImage, [20 60 119 39]);
    faceImage = imadjust(faceImage, stretchlim(eyesImage, [0.001 0.999]));
    faceImage = immultiply(im2double(faceImage), mask);
else
    faceImage = mask;
end
subplot(2, 3, [1 2 4 5]); subimage(inputImage); axis off;
subplot(2, 3, 3); subimage(eyesImage); axis off;
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
