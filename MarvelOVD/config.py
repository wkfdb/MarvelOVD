def add_detector_config(cfg):
    _C = cfg
    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.MODEL.ROI_BOX_HEAD.EMB_DIM = 512
    _C.MODEL.ROI_BOX_HEAD.LOSS_WEIGHT_BACKGROUND = 1.0

    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
