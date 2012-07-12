classdef Preprocessor < handle
%PREPROCESSOR Input image preprocessor.
%   Input images are preprocessed by first using a face detector to
%   find the subject face in the image, and then applying various
%   algorithms to normalize the size, rotation and illumination.
    
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
    
properties (SetAccess = private)
    InputImage;
    FaceImage;
    FaceMask;
    FaceDetector;
    EyesDetector;
end

methods
    function [ self ] = Preprocessor( )
        self = self@handle;
        self.FaceMask = facefactor.gaussianMask(200, 160);
        self.FaceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
        self.FaceDetector.MinSize = [170 170];
        self.FaceDetector.MergeThreshold = 7;
        self.EyesDetector = facefactor.EyesDetector();
    end

    function [ faceImage, confidence ] = step( self, inputImage )
        self.InputImage = inputImage;
        self.FaceImage = self.FaceMask;
        bbox = self.FaceDetector.step(inputImage);
        if ~isempty(bbox)
            faceBox = bbox(1, :);
            % Adjust for an apparent training error in the FaceDetector.
            faceBox(3:4) = faceBox(3:4) * 1.015;
            faceImage = imcrop(inputImage, faceBox);
            faceImage = imresize(faceImage, [200 NaN]);
            faceImage = imcrop(faceImage, [20 0 159 200]);
        else
            faceImage = self.FaceImage;
            confidence = 0;
            return;
        end
        inferImage = inputImage;
        [~, confidence, angle] = self.EyesDetector.step(faceImage);
        if angle ~= 0
            inferImage = imrotate(inferImage, double(angle));
            bbox = self.FaceDetector.step(inferImage);
            if ~isempty(bbox)
                faceBox = bbox(1, :);
                % Adjust for an apparent training error in the faceDetector.
                faceBox(3:4) = faceBox(3:4) * 1.015;
                faceImage = imcrop(inferImage, faceBox);
                faceImage = imresize(faceImage, [200 NaN]);
                faceImage = imcrop(faceImage, [20 0 159 200]);
            end
        end
        faceImage = immultiply(im2single(faceImage), self.FaceMask);
        faceImage = imadjust(faceImage, stretchlim(faceImage, [0.001 0.999]));
        self.FaceImage = faceImage;
    end
end
    
end

