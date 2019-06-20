import argparse
import tensorflow as tf
import os
import time
import logging
import model
import data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--validate_path", type=str, default="./data/ssepi")
    parser.add_argument("--save_path", type=str, default="./data/rec_dsepi")
    parser.add_argument("--tensorboard_path", type=str, default="./tensorboard")
    parser.add_argument("--shearlet_system_path", type=str, default="./model")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Debug logging output")
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    # set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # setup dataset
    create_dataset = data.create_image_dataset

    dataset_params = {
        "validate_path": args.validate_path,
        "save_path": args.save_path,
    }

    logging.info("Dataset params: %s" % str(dataset_params))

    def validate_fn():
        dataset_eval = create_dataset(train=False, params=dataset_params).batch(args.batch_size)
        eval_it = dataset_eval.make_one_shot_iterator()
        return eval_it.get_next()

    estimator = tf.estimator.Estimator(
        model_fn=model.create_model,
        params={
            "batch_size": args.batch_size,
            "tensorboard_dir": args.tensorboard_path,
            "shearlet_system_dir": args.shearlet_system_path,
            "num_output_channels": 3,
            "height": 128,
            "width": 608, 
            "alpha": 2,
            "niter": 30,
            "thmax": 2,
            "thmin": 0.02
        },
        config=tf.estimator.RunConfig(
            model_dir=args.tensorboard_path,
        )
    )

    predictions = estimator.predict(input_fn=validate_fn)

    for i, pred_dict in enumerate(predictions):
        save_name, im_rec = pred_dict["save_name"].decode("utf-8"), pred_dict["image"]
        im_rec = cv2.cvtColor(im_rec, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_name, im_rec)
        print("Image saved in", save_name)
        
    print('Required time: {:.3f} s'.format(time.time() - start) )
