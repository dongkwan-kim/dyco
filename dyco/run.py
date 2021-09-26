import hydra
from omegaconf import DictConfig


"""Codes are adopted from
    https://github.com/ashleve/lightning-hydra-template/blob/main/run.py"""


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    import logger_utils
    from train import train

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    logger_utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        logger_utils.print_config(config, resolve=True)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()