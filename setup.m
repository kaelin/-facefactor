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
ipp = facefactor.Preprocessor();

%% Recognition
rec = facefactor.Recognizer('rec-ex3.mat');
