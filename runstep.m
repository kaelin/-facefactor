%% Run step
inputImage = facefactor.fetchInputImage(cam);

%% Preprocess image
% tic;
faceImage = pp.step(inputImage);
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
