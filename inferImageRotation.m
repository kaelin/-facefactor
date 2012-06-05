function [ angle, confidence ] = inferImageRotation( eyesImage, eyesMask, engine, hax )
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
[CLt, MLt, ILt] = facefactor.clusterMSERRegions(regionsLt, 25, 3);
[CRt, MRt, IRt] = facefactor.clusterMSERRegions(regionsRt, 25, 3);

MLt
MRt

OLt = arrayfun(@(i) ellipsity(regionsLt(ILt == i).Axes), 1:length(MLt))
ORt = arrayfun(@(i) ellipsity(regionsRt(IRt == i).Axes), 1:length(MRt))

SLt = arrayfun(@(i) {std(regionsLt(ILt == i).Centroid, 0, 1)}, 1:length(MLt));
cell2mat(SLt')
SRt = arrayfun(@(i) {std(regionsRt(IRt == i).Centroid, 0, 1)}, 1:length(MRt));
cell2mat(SRt')

% Do inference on the evidence we observe for each candidate cluster
ELt = arrayfun(@inference, MLt, OLt, SLt)
ERt = arrayfun(@inference, MRt, ORt, SRt)

confLt = max(ELt);
pickLt = find(ELt == confLt);
if length(pickLt) ~= 1
    pickLt = find(OLt == min(OLt));
end
confRt = max(ERt);
pickRt = find(ERt == confRt);
if length(pickRt) ~= 1
    pickRt = find(ORt == min(ORt));
end

if length(pickLt) == 1 && length(pickRt) == 1
    angle = round(atan2(CRt(pickRt, 2) - CLt(pickLt, 2), CRt(pickRt, 1) - CLt(pickLt, 1)) * (180 / pi));
    confidence = min(ELt(pickLt), ERt(pickRt));
end

if nargin > 2
    axes(hax);
    subimage(1 - eyesImage * 0.9); hold all;
    % contour(eyesMask); hold all;
    plot(regionsLt); hold all;
    plot(regionsRt); hold all;
    if ~isempty(CLt)
        scatter(CLt(:, 1), CLt(:, 2), (MLt * 3).^1.7, 'Filled'); hold all;
        scatter(CLt(pickLt, 1), CLt(pickLt, 2), 'd'); hold all;
    end
    if ~isempty(CRt)
        scatter(CRt(:, 1), CRt(:, 2), (MRt * 3).^1.7, 'Filled'); hold all;
        scatter(CRt(pickRt, 1), CRt(pickRt, 2), 'd'); hold all;
    end
    hold off;
end

    function [ o ] = ellipsity( axes )
        o = min(axes(:, 1) ./ axes(:, 2));
    end

    function [ e ] = inference( M, O, S )
        S = cell2mat(S);
        evidence = cell(1, 5);
        evidence{2} = sum(M > [0 2 6]);
        evidence{3} = sum(O > [0 1.57 2 3]);
        evidence{4} = sum(S(1) > [-Inf 1 2]);
        evidence{5} = sum(S(2) > [-Inf 0.4 0.8]);
        eng = enter_evidence(engine, evidence);
        marginals = marginal_nodes(eng, 1);
        e = marginals.T(1);
    end

end
