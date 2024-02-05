import sys
import yaml
from osl_dynamics.config_api.batch import IndexParser, BatchTrain

if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    with open(config_path,'r') as file:
        config_batch = yaml.safe_load(file)
    index_parser = IndexParser(config_batch)
    config = index_parser.parse(index)
    batch_train = BatchTrain(config)
    batch_train.model_train()
