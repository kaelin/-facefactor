%% Camera
if ~exist('cam', 'var')
    cam = videoinput('macvideo', 3, 'YCbCr422_1280x720');
    cam.ROIPosition = [290 20 700 700];
    cam.ReturnedColorSpace = 'grayscale';
    cam.FramesPerTrigger = 1;
    cam.TriggerRepeat = Inf;
    triggerconfig(cam, 'manual');
    start(cam);
else
    disp('Using existing workspace cam.');
end

%% Face detection
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
faceDetector.MinSize = [170 170];
faceDetector.MergeThreshold = 7;

faceMask = facefactor.gaussianMask(200, 160);
eyesCrop = [20 58 119 39];
eyesMask = [facefactor.gaussianMask(eyesCrop(4) + 1, 50) ...
    zeros(eyesCrop(4) + 1, 20) ...
    facefactor.gaussianMask(eyesCrop(4) + 1, 50)];

%% Recognition
rec = facefactor.Recognizer('rec-ex3.mat');
