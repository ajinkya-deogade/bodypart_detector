import json

# do_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_205257/Annotations_And_Frames/Rawdata_20170303_205257_Frames_20170306_141211/Rawdata_20170303_205257_20170306_141211_Coordinates.json'
# bo_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_205257/Annotations_And_Frames/Rawdata_20170303_205257_20170309_150543_and_154054_Coordinates.json'

# do_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_210131/Annotations_And_Frames/Rawdata_20170303_210131_Frames_20170306_153534/Rawdata_20170303_210131_20170306_153534_Coordinates.json'
# bo_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/Gaussian/Rawdata_20170303_210131/Annotations_And_Frames/Rawdata_20170303_210131_Frames_20170309_174237/Rawdata_20170303_210131_20170309_174237_Coordinates.json'

do_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/InverseGaussian/Rawdata_20170303_201605/Annotations_And_Frames/Rawdata_20170303_201605_Frames_20170306_195005/Rawdata_20170303_201605_20170306_195005_Coordinates.json'
bo_train_annotation_file = '/Volumes/NewHD/Dropbox (CRG)/Tracker Development (Ajinkya)/MHDO_Tracking/data/Janelia_Q1_2017/20170303_experiments/MouthHook/InverseGaussian/Rawdata_20170303_201605/Annotations_And_Frames/Rawdata_20170303_201605_20170309_142501_102846_Coordinates.json'

with open(do_train_annotation_file) as fin_annotation:
    do_train_annotation = json.load(fin_annotation)

with open(bo_train_annotation_file) as fin_annotation:
    bo_train_annotation = json.load(fin_annotation)

for i in range(0, len(do_train_annotation['Annotations'])):
    if int(do_train_annotation['Annotations'][i]['FrameIndexVideo']) == int(bo_train_annotation['Annotations'][i]['FrameIndexVideo']):
        do_train_annotation['Annotations'][i]['FrameValueCoordinates'].extend(bo_train_annotation['Annotations'][i]['FrameValueCoordinates'])

output_file = do_train_annotation_file[:-5] + '_withBolwigs.json'
with open(output_file, 'w') as outfile:
    json.dump(do_train_annotation, outfile)