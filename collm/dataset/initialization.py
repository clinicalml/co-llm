import argparse
import logging

import datasets
import torch

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_model_ds", type=str, default="default")
    parser.add_argument("--deferral_model_ds", type=str, default="default")
    parser.add_argument("--init_method", type=str, default="default")
    parser.add_argument("--output_dir", type=str, default="default")
    parser.add_argument("--log_level", type=str, default="DEBUG")
    args = parser.parse_args()

    logger.setLevel(args.log_level)
    logger.debug("Base  df  (Small Model)  path : {}".format(args.training_model_ds))
    logger.debug("Deferral df (Large Model) path: {}".format(args.deferral_model_ds))
    logger.debug("Output path:                    {}".format(args.output_dir))
    logger.debug("Init method: {}".format(args.init_method))

    ds1 = datasets.load_from_disk(args.deferral_model_ds)
    ds2 = datasets.load_from_disk(args.training_model_ds)
    assert len(ds1) == len(ds2)

    all_more_confidence = []
    for example_id in range(len(ds1)):
        is_reference_max_deferral = torch.isclose(
            ds1[example_id]["reference_log_probs"],
            ds1[example_id]["reference_max_prob"],
        )
        is_reference_max_training = torch.isclose(
            ds2[example_id]["reference_log_probs"],
            ds2[example_id]["reference_max_prob"],
        )

        if args.init_method == "a":
            # When the deferral model is correct, while the training model is not
            more_confident = is_reference_max_deferral > is_reference_max_training
            more_confident = more_confident.long()
            ## Some additional note for why this is correct
            # more_confident = torch.logical_and(
            #     torch.logical_or(is_reference_max_deferral, is_reference_max_training), ~is_reference_max_training
            # )
            # more_confident = more_confident.long()
            # a = torch.logical_and(
            #     torch.logical_or(is_reference_max_deferral, is_reference_max_training), ~is_reference_max_training
            # )
            # b = is_reference_max_deferral > is_reference_max_training
            # assert torch.allclose(a, b)

        elif args.init_method == "b":
            # When the training model is not correct.
            more_confident = (~is_reference_max_training).long()
        else:
            raise NotImplementedError("Unknown init method {}".format(args.init_method))

        # We want to mask out the log probs in user conversations, similar to what
        # the tulu paper does
        input_mask = ds1[example_id]["labels"][1:] == -100
        more_confident[input_mask] = -100
        all_more_confidence.append(more_confident)

    ds3 = ds1.remove_columns(["reference_log_probs"]).add_column(
        "reference_log_probs", [ele.long().tolist() for ele in all_more_confidence]
    )
    ds3.save_to_disk(args.output_dir)

    no_zero = (torch.cat(all_more_confidence) == 1).float().mean()
    print("Average non-zero ratio: {}".format(no_zero))
