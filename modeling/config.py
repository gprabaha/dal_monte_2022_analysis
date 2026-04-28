import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--model_path", type=str, default="checkpoints/social_mrnn")
    parser.add_argument("--experiment", type=str, default="")

    return parser