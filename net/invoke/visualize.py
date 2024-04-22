import invoke


@invoke.task
def maps_model_predictions_in_order(_context, config_path):

    import glob
    import os
    import sys

    import box
    import cv2
    import PIL
    import torch
    import tqdm

    import data.base_dataset
    import data.aligned_dataset
    import models
    import options.test_options
    import util.util

    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    import icecream

    sys.argv = [
        "whatever",
        "--dataroot", "/data/maps",
        "--name", "maps_pix2pix",
        "--dataset_mode", "aligned",
        "--model", "pix2pix",
        "--netG", "unet_256",
        "--norm", "batch"
    ]

    opt = options.test_options.TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    all_twin_images_paths = sorted(glob.glob(
        pathname=os.path.join(config.maps_dataset.test_data_dir, "*.jpg")))

    model = models.create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    images_logger = net.utilities.get_images_logger(
        path=config["logging_path"],
        images_directory=os.path.join(os.path.dirname(config["logging_path"]), "images"),
        images_html_path_prefix="images"
    )

    for twin_image_path in tqdm.tqdm(all_twin_images_paths[:50]):

        data_map = data.aligned_dataset.AlignedDataset.get_pair_image_data_map(
            pair_image_path=twin_image_path,
            opt=opt,
            input_nc=3,
            output_nc=3
        )

        data_map["A"] = torch.unsqueeze(data_map["A"], dim=0)
        data_map["B"] = torch.unsqueeze(data_map["B"], dim=0)

        model.set_input(data_map)  # unpack data from data loader
        model.test()

        visuals = model.get_current_visuals()  # get image results

        images_logger.log_images(
            twin_image_path,
            images=[
                cv2.cvtColor(util.util.tensor2im(visuals[key]), cv2.COLOR_RGB2BGR)
                for key in ["real_A", "fake_B", "real_B"]
            ]
        )
