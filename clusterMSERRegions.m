function [ C, M, IDX, D ] = clusterMSERRegions( regions, distance, max_k )
%CLUSTERMSERREGIONS Use K-means to cluster MSER regions.
%   This function uses the centroids of a batch of detected MSER regions to
%   partition them into from one to max_k clusters. It stops trying to find
%   clusters when the partitioning satisfies the specified distance
%   parameter.

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

C = [];
M = [];
IDX = [];
D = [];

if regions.Count == 0
    return;
end

if regions.Count == 1
    C = regions.Centroid;
    M = 1;
    IDX = 1;
    D = 0;
    return;
end

try
    for k = 1:max_k
        [IDX, C, sumd, D] = kmeans(regions.Centroid, k);
        if max(sumd) <= distance
            break;
        end
    end
    M = arrayfun(@(i) sum(IDX == i), 1:k);
catch ME
    % Error using kmeans/batchUpdate (line 376)
    % Empty cluster created at iteration 2.
    disp(ME.message);
    [IDX, C, ~, D] = kmeans(regions.Centroid, k - 1);
end

end
