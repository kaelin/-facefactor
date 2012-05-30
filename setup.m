%% Camera
if ~exist('cam')
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

mask = facefactor.gaussianMask(200, 160);
eyesMask = [facefactor.gaussianMask(40, 50) zeros(40, 20) facefactor.gaussianMask(40, 50)];

%% Recognition
rec = facefactor.Recognizer('rec-ex3.mat');
