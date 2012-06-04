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

%% Rotation inference
eyesCrop = [20 58 119 39];
eyesMask = facefactor.gaussianMask(eyesCrop(4) + 1, 70);
eyesMask = [eyesMask(:, 11:60) zeros(eyesCrop(4) + 1, 20) eyesMask(:, 11:60)];

DAG = false(5);
DAG(1, 2:5) = true;
bnet = mk_bnet(DAG, [2 3 3 3 3]);
bnet.CPD{1} = tabular_CPD(bnet, 1, [.5  .5]);
bnet.CPD{2} = tabular_CPD(bnet, 2, [.1  .345   .45 .355   .45 .3]);
bnet.CPD{3} = tabular_CPD(bnet, 3, [.01 .33    .79 .66    .2  .01]);
bnet.CPD{4} = tabular_CPD(bnet, 4, [.49 .33333 .5  .33333 .01 .33333]);
bnet.CPD{5} = tabular_CPD(bnet, 5, [.5  .33333 .49 .33333 .01 .33333]);
engine = jtree_inf_engine(bnet);

%% Recognition
rec = facefactor.Recognizer('rec-ex3.mat');
