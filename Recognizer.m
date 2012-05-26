classdef Recognizer < handle
%RECOGNIZER Facial recognition using the eigenfaces algorithm.
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

properties
    LabelNames = {};
    KnownSet = repmat(struct('label', 0, 'image', []), 1, 0);
    Training = struct('isvalid', false, 'mean', [], 'eigenfaces', [], ...
        'indexlabel', [], 'labelcoord', [], 'rawepsilon', [], 'epsilon', Inf);
end

methods
    function [ self ] = Recognizer( filename )
        self = self@handle;
        if nargin > 0
            matf = matfile(filename);
            self.LabelNames = matf.labelNames;
            self.KnownSet = matf.knownSet;
            self.Training = matf.training;
        end
    end

    function addKnown( self, labelname, image )
        set = self.KnownSet;
        set(end + 1).label = self.labelForName(labelname);
        set(end).image = image;
        self.KnownSet = set;
    end

    function [ label ] = labelForName( self, labelname )
        tmp = arrayfun(@(name) isequal(name{1}, labelname), self.LabelNames);
        label = find(tmp);
        if (isempty(label))
            self.LabelNames(end + 1) = {labelname};
            label = length(self.LabelNames);
        end
    end

    function saveState( self, filename )
        matf = matfile(filename, 'Writable', true);
        matf.labelNames = self.LabelNames;
        matf.knownSet = self.KnownSet;
        matf.training = self.Training;
    end

    function [ label, score ] = step( self, image )
        X = image(:) - self.Training.mean; % Normalize
        W = self.Training.eigenfaces' * X; % Compute coordinates in eigenfaces space
        [label, score] = self.assignLabelAndScore(W);
    end

    function [ stats ] = train( self, k )
        self.Training.isvalid = false;
        self.Training.epsilon = Inf;
        tic;
        nlabels = length(self.LabelNames);
        fprintf('Training with %i labels on %i images (k = %i):\n', ...
            nlabels, length(self.KnownSet), k);
        [T, V, labelT, labelV] = self.partitionKnownSet(k);
        self.Training.mean = mean(T, 2);
        % self.Training.mean = mean([T V], 2);
        T = T - repmat(self.Training.mean, 1, size(T, 2)); % Normalize
        L = T' * T;
        eigs_k = min(30, size(T, 2) - 1);
        fprintf('Computing %i eigenfaces dimensions.\n', eigs_k);
        [E, ~] = eigs(L, eigs_k);
        self.Training.eigenfaces = T * E;  % Compute eigenfaces
        WT = self.Training.eigenfaces' * T; % Compute coordinates in eigenfaces space
        % Here's where the implementation options begin to drastically
        % expand. At this point we have up to k distinct coordinates
        % for each label in the training set. The original eigenface
        % paper recommends computing an "average" coordinate for each
        % label. We could instead choose to keep each distinct
        % coordinate for each label, which may avoid some hypothetical
        % ambiguity introduced by averaging-- at the cost of definitely
        % increasing the number of scores we ultimately have to
        % compute. We could also opt for a hybrid approach, averaging
        % coordinates that are already "close" into clusters that still
        % account for significant variations between images assigned
        % the same label.
        self.finishTrainingBasicAverageCoordinate(labelT, WT);
        % Now we run through the validation set and see if we can learn
        % a suitable final value for epsilon.
        V = V - repmat(self.Training.mean, 1, size(V, 2)); % Normalize
        WV = self.Training.eigenfaces' * V; % Compute coordinates in eigenfaces space
        [labelA, scoreA] = self.assignLabelAndScore(WV);
        % In the interest of starting simple, initially we'll base
        % epsilon solely on the measured variance in the training set.
        epsilon = quantile(self.Training.rawepsilon, .75)
        elideA = scoreA > epsilon;
        self.Training.epsilon = epsilon;
        self.Training.isvalid = true;
        stats.T = T;
        stats.labelT = labelT;
        % stats.L = L;
        % stats.E = E;
        stats.WT = WT;
        stats.V = V;
        stats.labelV = labelV;
        stats.WV = WV;
        stats.labelA = labelA;
        stats.scoreA = scoreA;
        stats.elideA = elideA;
        toc;
    end
end

methods (Access = private)
    function [ label, score ] = assignLabelAndScore( self, W )
        ncoord = size(self.Training.labelcoord, 2);
        n = size(W, 2);
        label = zeros(1, n);
        score = NaN(1, n);
        for i = 1:n
            scores = arrayfun(@(j) norm(W(:, i) - self.Training.labelcoord(:, j)), 1:ncoord);
            score(i) = min(scores);
            if score(i) < self.Training.epsilon
                label(i) = self.Training.indexlabel(scores == score(i));
            end
        end
    end

    function finishTrainingBasicAverageCoordinate( self, labelT, WT )
        % Compute the "average" coordinate for each label.
        nlabels = length(self.LabelNames);
        eigs_k = size(WT, 1);
        self.Training.indexlabel = 1:nlabels; % No indirection necessary
        self.Training.labelcoord = zeros(eigs_k, nlabels);
        self.Training.rawepsilon = NaN(1, nlabels);
        for i = 1:nlabels
            X = WT(:, labelT == i);
            meanX = mean(X, 2);
            highscore = max(arrayfun(@(j) norm(X(:, j) - meanX), 1:size(X, 2)));
            self.Training.labelcoord(:, i) = meanX;
            if highscore > 0
               self.Training.rawepsilon(:, i) = highscore * 1.001;
            end
        end
    end

    function [ T, V, labelT, labelV ] = partitionKnownSet( self, k )
        nlabels = length(self.LabelNames);
        T = [];
        V = [];
        labelT = [];
        labelV = [];
        for i = 1:nlabels
            selected = [self.KnownSet.label] == i;
            fprintf('    %-20s  %2i image(s)\n', ...
                self.LabelNames{i}, sum(selected));
            indexes = find(selected);
            assert(~isempty(indexes), 'KnownSet contains no images for label %i', i);
            for j = 1:min(k, length(indexes))
                T = [T self.KnownSet(indexes(j)).image(:)];
                labelT = [labelT i];
            end
            for j = j + 1:length(indexes)
                V = [V self.KnownSet(indexes(j)).image(:)];
                labelV = [labelV i];
            end
        end
    end
end
    
end
