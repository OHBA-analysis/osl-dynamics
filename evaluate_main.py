import sys
import yaml
from osl_dynamics.config_api.batch import IndexParser, BatchTrain, batch_check

def main(index,config_path,analysis_config_path=None):
    '''
    This is the main function of all the analysis.
    Parameters
    ----------
    index: int
        index == -1 represents post training checkk or analysis.
    config_path: str
        where read the config path.
    analysis_config_path: str
        if analysis_config_path is not None, and index == -1
        then implement the analysis code.
    '''
    with open(config_path, 'r') as file:
        config_batch = yaml.safe_load(file)
    if index > -1:
        if isinstance(config_batch,list):
            with open(f'{config_batch[index]}batch_config.yaml','r') as file:
                config = yaml.safe_load(file)
        else:
            index_parser = IndexParser(config_batch)
            config = index_parser.parse(index)
        batch_train = BatchTrain(config)
        batch_train.model_train()
    else:
        # Step 1: batch check whether training is successful
        #          return the list where training is not successful
        batch_check(config_batch)

        # Step 2: if analysis_config_path is not None, implement the analayis code.
        if analysis_config_path is not None:
            pass




if __name__ == '__main__':
    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]

    if len(sys.argv) > 3:
        analysis_config_path = sys.argv[3]
    else:
        analysis_config_path = None

    main(index,config_path,analysis_config_path)


