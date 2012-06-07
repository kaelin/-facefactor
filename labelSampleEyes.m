function labelSampleEyes(  )
%LABELSAMPLEEYES Summary of this function goes here
%   Detailed explanation goes here

%   Copyright (C) 2012 Kaelin Colclasure

%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%   
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%   
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Example of editing a mislabelled sample ('>>' is the MATLAB command prompt):
% 
% ---8<---
% inferAngle =
% 
%    -11
% 
% inferConfidence =
% 
%     0.9558
% 
% sample = 
% 
%     [         1]    [         1]    [         2]
%     [         8]    [         7]    [         9]
%     [    1.3793]    [    1.2842]    [    2.1983]
%     [2x1 single]    [2x1 single]    [2x1 single]
%     [   36.9647]    [   29.8442]    [   38.9447]
% 
% Accept assigned labels in `sample` above? y/n [y]: n
% Saved `sample` to Workspace for manual editing.
% >> sample(1, :) = [{1} {2} {1}]
% 
% sample = 
% 
%     [         1]    [         2]    [         1]
%     [         8]    [         7]    [         9]
%     [    1.3793]    [    1.2842]    [    2.1983]
%     [2x1 single]    [2x1 single]    [2x1 single]
%     [   36.9647]    [   29.8442]    [   38.9447]
% 
% >> samples = [samples sample];
% --->8---

cam = evalin('base', 'cam');
faceDetector = evalin('base', 'faceDetector');
faceMask = evalin('base', 'faceMask');
eyesCrop = evalin('base', 'eyesCrop');
eyesMask = evalin('base', 'eyesMask');
engine = evalin('base', 'engine');

try
    samples = evalin('base', 'samples');
    assert(size(samples, 1) == 5, 'Wrong-sized samples!');
catch ME
    reply = input('Create new `samples` variable in Workspace? y/n [y]: ', 's');
    if isempty(reply)
        reply = 'y';
    end
    if reply ~= 'y'
        return;
    end
    samples = cell(5, 0);
end

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
    return;
end

eyesImage = im2double(eyesImage);

%% Infer image rotation
figure(1); clf;
inferImage = inputImage;
[inferAngle, inferConfidence, sample] = facefactor.inferImageRotation(eyesImage, eyesMask, engine, gca)
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

%% Prompt for correct labels
reply = input('Accept assigned labels in `sample` above? y/n [y]: ', 's');
if isempty(reply)
    reply = 'y';
end
if reply ~= 'y'
    disp('Saved `sample` to Workspace for manual editing.');
    assignin('base', 'sample', sample);
    return;
end
samples = [samples sample];

%% Save samples to Workspace
assignin('base', 'samples', samples);

end
