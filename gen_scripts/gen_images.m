%% Generate light, medium and dark composite images
clear all
load('yale_dark.mat')

composite_light = [];
for i=1:10
    composite_light = [composite_light reshape(dictSetSmall(:,i*30),32,32)];
end
figure(338);clf;
imshow(composite_light, [])

composite_dark = [];
for i=1:10
    composite_dark = [composite_dark reshape(testSetSmall(:,i*30),32,32)];
end
figure(328);clf;
imshow(composite_dark, [])

load('yale.mat')
composite_medium = [];
for i=1:10
    composite_medium = [composite_medium reshape(testSetSmall(:,i*27),32,32)];
end
figure(348);clf;
imshow(composite_medium, [])