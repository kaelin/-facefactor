classdef EyesDetector < handle
%EYESDETECTOR Estimate eye locations using a naive Bayes classifier.
%   Documentation forthcoming...
    
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

properties (SetAccess = immutable)
    Crop = [20 58 119 39];
    Mask;
    Bnet;
end

properties (Access = private)
    Engine;
end

properties (SetAccess = private)
    Image;
    MSERRegions;
    Position;
    Sample;
    Score;
    Side;
    Pick;
end

methods
    function [ self ] = EyesDetector( )
        self = self@handle;
        eyesMask = facefactor.gaussianMask(self.Crop(4) + 1, 70);
        self.Mask = [eyesMask(:, 11:60) zeros(self.Crop(4) + 1, 20) eyesMask(:, 11:60)];
        matf = matfile('+facefactor/eye-bnet-v1.mat');
        self.Bnet = matf.bnet;
        self.Engine = jtree_inf_engine(self.Bnet);
    end
    
    function plot( self, hax )
        if nargin > 1
            axes(hax);
        end
        subimage(1 - self.Image * 0.9); hold all;
        % contour(self.Mask);
        plot(self.MSERRegions);
        pts = double(self.Position);
        for i = 1:length(self.Side)
            x = pts(i, 1);
            y = pts(i, 2);
            if self.Pick(i)
                color = 'm';
            else
                color = 'w';
            end
            if self.Side(i) == 1
                text(x, y, ['\leftarrow  ' num2str(self.Score(i), '%.3f')], ...
                    'HorizontalAlignment', 'left', 'Color', color, ...
                    'BackgroundColor', 'k');
            else
                text(x, y, [num2str(self.Score(i), '%.3f') '\rightarrow'], ...
                    'HorizontalAlignment', 'right', 'Color', color, ...
                    'BackgroundColor', 'k');
            end
        end
        hold off;
    end
    
    function [ positions, confidence, angle ] = step( self, faceImage )
        self.Image = im2double(imcrop(faceImage, self.Crop));
        self.adjustImage();
        
        % Detect MSER features
        regionsB4 = detectMSERFeatures(self.Image, ...
            'RegionAreaRange', [20 350], 'MaxAreaVariation', 0.25);

        % Postprocess and cluster MSER features
        regions = regionsB4;
        levels = facefactor.sampleMSERRegions(self.Image, regions);
        regions = facefactor.selectMSERRegions(regions, levels < 0.7);
        self.MSERRegions = regions; % Record for plot
        sides = arrayfun(@(x) sign(x - 60), regions.Centroid(:, 1));
        regionsLt = facefactor.selectMSERRegions(regions, sides == -1);
        regionsRt = facefactor.selectMSERRegions(regions, sides == 1);
        [CLt, MLt, ILt] = facefactor.clusterMSERRegions(regionsLt, 25, 3);
        [CRt, MRt, IRt] = facefactor.clusterMSERRegions(regionsRt, 25, 3);
        self.Position = [CLt; CRt];
        self.Side = ones(1, length(MLt) + length(MRt)) * 2;
        self.Side(1:length(MLt)) = 1;
        
        % Bail out unless we have candidates for both sides
        if isempty(MLt) || isempty(MRt)
            positions = [];
            confidence = 0;
            angle = 0;
            return;
        end
        
        % Sample clustered features
        self.Sample = cell(5, length(MLt) + length(MRt));
        self.Sample(2, :) = arrayfun(@(m) {m}, [MLt MRt]);
        OLt = arrayfun(@(i) ellipsity(regionsLt(ILt == i).Axes), 1:length(MLt));
        ORt = arrayfun(@(i) ellipsity(regionsRt(IRt == i).Axes), 1:length(MRt));
        self.Sample(3, :) = arrayfun(@(o) {o}, [OLt ORt]);
        SLt = arrayfun(@(i) {std(regionsLt(ILt == i).Centroid, 0, 1)}, 1:length(MLt));
        SRt = arrayfun(@(i) {std(regionsRt(IRt == i).Centroid, 0, 1)}, 1:length(MRt));
        self.Sample(4, :) = arrayfun(@(s) {s{1}'}, [SLt SRt]);
        self.Sample(5, :) = ...
            arrayfun(@(i) {norm(self.Position(i, :) - [60 20])}, 1:size(self.Position, 1));
        
        % Run inference and pick positions
        self.computeScores();
        [positions, confidence, angle] = self.pickPositions();
        
            function [ o ] = ellipsity( axes )
                o = min(axes(:, 1) ./ axes(:, 2));
            end
    end
end

methods (Access = private)
    function adjustImage( self )
        eyesBlank = mean(mean(self.Image));
        self.Image = self.Image - eyesBlank;
        self.Image = self.Image .* self.Mask;
        self.Image = self.Image + 0.999; % eyesBlank;
        self.Image = imadjust(self.Image);
    end
    
    function computeScores( self )
        evidence = cell(1, 5);
        self.Score = zeros(1, size(self.Sample, 2));
        for i = 1:size(self.Sample, 2)
            evidence(2:5) = self.Sample(2:5, i);
            marginal = marginal_nodes(enter_evidence(self.Engine, evidence), 1);
            self.Score(i) = marginal.T(1);
        end
    end
    
    function [ positions, confidence, angle ] = pickPositions( self )
        ELt = self.Score(self.Side == 1);
        ERt = self.Score(self.Side == 2);
        confLt = max(ELt);
        pickLt = find(ELt == confLt, 1);
        confRt = max(ERt);
        pickRt = find(ERt == confRt, 1);
        positions = [self.Position(pickLt, :); self.Position(pickRt+length(ELt), :)];
        confidence = min(confLt, confRt);
        angle = round(atan2( ...
            positions(2, 2) - positions(1, 2), ...
            positions(2, 1) - positions(1, 1)) * (180 / pi));
        
        % Fix positions for inputImage
        positions = positions + repmat(self.Crop(1:2), 2, 1);
        
        % Record picks
        self.Pick = false(1, length(self.Score));
        self.Pick([pickLt pickRt+length(ELt)]) = true;
        self.Sample(1, :) = {2};
        self.Sample(1, self.Pick) = {1};
    end
end

end
