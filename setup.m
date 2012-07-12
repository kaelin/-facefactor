%% Camera
if ~exist('cam', 'var')
    cam = videoinput('macvideo', 4, 'YCbCr422_1280x720');
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

%% Rotation inference
eyesDetector = facefactor.EyesDetector;

%% Recognition
rec = facefactor.Recognizer('rec-ex3.mat');
