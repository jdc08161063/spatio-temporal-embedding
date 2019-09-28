MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

ID_FILTER = 5
N_CLASSES = 1
COST_THRESHOLD = 0.5
NUSCENES_ROOT = ''
SENSOR = 'CAM_FRONT'

MAX_INSTANCES = 32
MAX_INSTANCES_SCENE = 256

# Meanshfit bandwidth
BANDWIDTH = 1.0
# Causal clustering cost threshold
CLUSTERING_COST_THRESHOLD = 1.5
# Filter detections containing less than MIN_PIXEL_THRESHOLD pixels
MIN_PIXEL_THRESHOLD = 200
# How long to keep old centers
CLUSTER_MEAN_LIFE = 10
