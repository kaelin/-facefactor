function labelSampleEyes(  )
%LABELSAMPLEEYES Utility for hand-labeling eye detector training samples.
%   This function is intended for use from the MATLAB command prompt, and
%   it depends on several variables being set in the base workspace. It
%   modifies the `samples` variable in the base workspace, so use it with
%   due caution.

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
eyesDetector = evalin('base', 'eyesDetector');

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
else
    faceImage = faceMask;
    return;
end

%% Infer image rotation
inferImage = inputImage;
[~, confidence, angle] = eyesDetector.step(faceImage);
figure(1); clf; eyesDetector.plot();
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
faceImage = immultiply(im2double(faceImage), faceMask);
faceImage = imadjust(faceImage, stretchlim(faceImage, [0.001 0.999]));
sample = eyesDetector.Sample

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
