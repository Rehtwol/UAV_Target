# UAV_Target
Targetting for the Gryphon UAV

Targetting code including letter recognition for the Gryphon UAV.
OCVTargets just picks inages from the camera and does basic blob detection which is to be used for manoeuvers.
LetterDetect runs a KNN system for detecting the letter contained in the target. This requires flattened_images.txt and classifications.txt.
