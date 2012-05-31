function [ samples ] = sampleMSERRegions( IMG, regions )
%SAMPLEMSERREGIONS Sample a representative pixel of each MSER region.
%   This function collects a list of sample pixel values. This can be used,
%   for example, to distinguish between lighter and darker MSER regions.

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

samples = arrayfun(@extractSample, regions.PixelList);

    function [ sample ] = extractSample( pixels )
        points = pixels{1};
        sample = IMG(points(1, 2), points(1, 1));
    end

end
