import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.build import build_detection_train_loader

from MarvelOVD.config import add_detector_config
from MarvelOVD.data.dataset_mapper_ssod import DatasetMapperSSOD as DatasetMapper

# from VL_PLM.data.dataset_mapper import DatasetMapperVLPLM as DatasetMapper
from MarvelOVD.data.augumentation import build_lsj_aug
from MarvelOVD.evaluation.coco_evaluation import COCO_evaluator

logger = logging.getLogger("detectron2")


def do_test(cfg, model, data_loader):
    results = OrderedDict()
    
    for dataset_name in cfg.DATASETS.TEST:
        
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))

        metadata = MetadataCatalog.get(dataset_name)
        distributed = comm.get_world_size() > 1
        if distributed:
            model.module.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)
        else:
            model.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)

        evaluator_type = metadata.evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco' in evaluator_type:
            evaluator = COCO_evaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False, use_lsj=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    # try:
    #     start_iter = (
    #             checkpointer.resume_or_load(
    #                 cfg.OUTPUT_DIR, resume=True,
    #             ).get("iteration", -1) + 1
    #     )
    # except:
    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
        ).get("iteration", -1) + 1
    )
    print(start_iter)
    # start_iter = (
    #     checkpointer.resume_or_load(
    #         cfg.MODEL.WEIGHTS, resume=resume,
    #     ).get("iteration", -1) + 1
    # )
    
    
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    mapper = DatasetMapper(cfg, True)
    # if use_lsj:
    #     mapper.augmentations = build_lsj_aug(image_size=1024)
    #     mapper.recompute_boxes = True

    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    test_data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    distributed = comm.get_world_size() > 1
    if distributed:
        model.module.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)
    else:
        model.roi_heads.box_predictor.set_class_embeddings(metadata.class_emb_mtx)

    logger.info("Starting training from iteration {}".format(start_iter))

    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):

            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            storage.step()
            if iteration<=500:
                loss_dict = model(data,"burn-in")
            else:
                loss_dict = model(data)
            # loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                                 for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and iteration % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter
                    and iteration != 0
            ):
                do_test(cfg, model, test_data_loader)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                    (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            iteration = iteration + 1
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detector_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
        test_data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        return do_test(cfg, model, test_data_loader)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume, use_lsj=args.use_lsj)
    return
    # return do_test(cfg, model, test_data_loader)


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args.add_argument('--use_lsj', action='store_true', default=False)
    args = args.parse_args()
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
