
def HMM_training(dataset,n_states,n_channels,save_dir):
    from osl_dynamics.models.hmm import Config, Model
    # Create a config object
    config = Config(
        n_states=n_states,
        n_channels=n_channels,
        sequence_length=600,
        learn_means=True,
        learn_covariances=True,
        batch_size=32,
        learning_rate=1e-3,
        n_epochs=40,
    )
    
    # Initiate a Model class and print a summary
    model = Model(config)
    model.summary()
    
    # Initialization
    init_history = model.random_state_time_course_initialization(dataset, n_epochs=1, n_init=3)
    
    # Full training
    history = model.fit(training_data)
    model.save(save_dir)