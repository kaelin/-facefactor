function [ M ] = gaussianMask( m, n )
%GAUSSIANMASK Generate a gaussian mask matrix with the specified size.
%   This mask is used to de-emphasize features at the borders of an image,
%   the idea being to concentrate on the features in the center of a
%   person's face.

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

halfm = floor(m / 2);
halfn = floor(n / 2);

x = -(halfn - 1):halfn;
y = -(halfm - 1):halfm;

x = x .* (2 / halfn);
y = y .* (2 / halfm);

[X, Y] = meshgrid(x, y);

M = mvnpdf([X(:) Y(:)]);
M = M / max(M);
M = reshape(M, length(y), length(x));

end
