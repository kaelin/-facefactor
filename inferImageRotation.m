function [ angle, confidence ] = inferImageRotation( eyesImage, eyesMask, hax )
%INFERINPUTIMAGEROTATION Summary of this function goes here
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

angle = 0;
confidence = 0;

eyesBlank = mean(mean(eyesImage));
eyesImage = eyesImage - eyesBlank;
eyesImage = eyesImage .* eyesMask;
eyesImage = eyesImage + 0.999; % eyesBlank;
eyesImage = imadjust(eyesImage);

% Detect MSER features
regionsB4 = detectMSERFeatures(eyesImage, 'RegionAreaRange', [20 350], 'MaxAreaVariation', 0.25);

% Postprocess and cluster MSER features
regions = regionsB4;
levels = facefactor.sampleMSERRegions(eyesImage, regions);
regions = facefactor.selectMSERRegions(regions, levels < 0.7);
sides = arrayfun(@(x) sign(x - 60), regions.Centroid(:, 1));
regionsLt = facefactor.selectMSERRegions(regions, sides == -1);
regionsRt = facefactor.selectMSERRegions(regions, sides == 1);
[CLt, MLt] = facefactor.clusterMSERRegions(regionsLt, 25, 3);
[CRt, MRt] = facefactor.clusterMSERRegions(regionsRt, 25, 3);

if length(MLt) == 1 && length(MRt) == 1
    angle = round(atan2(CRt(1, 2) - CLt(1, 2), CRt(1, 1) - CLt(1, 1)) * (180 / pi));
    confidence = .95;
end

if nargin > 2
    axes(hax);
    subimage(1 - eyesImage * 0.9); hold all;
    contour(eyesMask); hold all;
    plot(regionsLt); hold all;
    plot(regionsRt); hold all;
    if ~isempty(CLt)
        scatter(CLt(:, 1), CLt(:, 2), (MLt * 3).^1.7, 'Filled'); hold all;
    end
    if ~isempty(CRt)
        scatter(CRt(:, 1), CRt(:, 2), (MRt * 3).^1.7, 'Filled'); hold all;
    end
    hold off;
end

end
