


def parse_index(index:int,models:list,list_channels:list,list_states:list):
    '''
    This function is used in the array job. Given an index,
    return the model, n_channels, n_states accordingly
    
    Parameters:
    index: (int) the input index in the array job
    models: (list) the model list
    list_channels: (list) the n_channel list
    list_states: (list) the n_state list
    
    Returns:
        tuple: A tuple containing the following
            - model (string): The model to use
            - n_channels (int): The number of channels to use
            - n_states (int): The number of states to use
    '''
    
    N_n_channels = len(list_channels)
    N_n_states = len(list_states)
    
    model = models[index // (N_n_channels * N_n_states)]
    index = index % (N_n_channels * N_n_states)
    
    n_channels = list_channels[index // N_n_states]
    n_states = list_states[index % N_n_states]
    
    return model, n_channels, n_states
    '''
    if index >= 30:
            model = models[1]
            index -= 30
        else:
            model = models[0]
    n_channels = list_channels[index // 5]
    n_states = list_states[index % 5]
    '''