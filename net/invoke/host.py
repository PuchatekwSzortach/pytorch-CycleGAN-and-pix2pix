"""
Module with docker commands
"""

import invoke


@invoke.task
def build_app_container(context):
    """
    Build app container

    :param context: invoke.Context instance
    """

    command = (
        "DOCKER_BUILDKIT=1 docker build "
        "--tag puchatek_w_szortach/pytorch_pix2pix:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def run(context, config_path):
    """
    Run app container

    Args:
        context (invoke.Context): invoke context instance
        config_path (str): path to configuration file
    """

    import os

    import box

    import net.host.utilities

    config = box.Box(net.host.utilities.read_yaml(config_path))

    # Define run options that need a bit of computations
    run_options = {
        # Use gpu runtime if host has cuda installed
        "gpu_capabilities": "--gpus all" if "/cuda/" in os.environ["PATH"] else ""
    }

    os.makedirs(config.data_dir_on_host, exist_ok=True)
    os.makedirs(config.logging_output_directory_on_host, exist_ok=True)

    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app "
        f"-v {config.logging_output_directory_on_host}:{os.path.dirname(config.logging_path)} "
        f"-v {os.path.abspath(config.data_dir_on_host)}:{config.data_dir} "
        "puchatek_w_szortach/pytorch_pix2pix:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)
