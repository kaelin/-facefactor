function [ inputImage ] = fetchInputImage( cam )
%FETCHINPUTIMAGE Fetch an image from the access control camera
%   This function retrieves a single grayscale image from the access
%   control camera.

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

cam.trigger();
inputImage = cam.getdata();

end
