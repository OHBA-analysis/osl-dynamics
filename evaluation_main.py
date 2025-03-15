import sys


def main(index,config_path):
    from osl_dynamics.config_api.batch import IndexParser
    '''
    Main function of all the analysis.
    Parameters
    ----------
    index: int
        index == -1 represents initialisation of the training
    config_path: str
        Path to the configuration file.
    '''

    index_parser = IndexParser(config_path)
    if index == -1:
        index_parser.make_list()
    else:
        index_parser.parse(index)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_main.py <index> <config_path>")
        sys.exit(1)

    index = int(sys.argv[1]) - 1
    config_path = sys.argv[2]
    main(index,config_path)