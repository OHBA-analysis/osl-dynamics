"""Base class for observation models.

"""

from tensorflow.keras import optimizers
from dynemo.inference import initializers
from dynemo.models.base import Base
from dynemo.utils.misc import replace_argument


class ObservationModelBase(Base):
    """Base class for observation models.

    Parameters
    ----------
    config : dynemo.models.Config
    """

    def __init__(self, config):
        # The base class will build and compile the model
        Base.__init__(self, config)

    def compile(self, optimizer=None):
        """Wrapper for the standard keras compile method."""
        if optimizer is None:
            optimizer = optimizers.get(
                {
                    "class_name": self.config.optimizer.lower(),
                    "config": {
                        "learning_rate": self.config.learning_rate,
                        "clipnorm": self.config.gradient_clip,
                    },
                }
            )
        self.model.compile(optimizer)

    def fit(
        self,
        *args,
        use_tqdm=False,
        tqdm_class=None,
        use_tensorboard=None,
        tensorboard_dir=None,
        save_best_after=None,
        save_filepath=None,
        **kwargs,
    ):
        """Wrapper for the standard keras fit method.

        Adds callbacks and then trains the model.

        Parameters
        ----------
        use_tqdm : bool
            Should we use a tqdm progress bar instead of the usual output from
            tensorflow.
        tqdm_class : tqdm
            Class for the tqdm progress bar.
        use_tensorboard : bool
            Should we use TensorBoard?
        tensorboard_dir : str
            Path to the location to save the TensorBoard log files.
        save_best_after : int
            Epoch number after which we should save the best model. The best model is
            that which achieves the lowest loss.
        save_filepath : str
            Path to save the best model to.

        Returns
        -------
        history
            The training history.
        """
        if use_tqdm:
            args, kwargs = replace_argument(self.model.fit, "verbose", 0, args, kwargs)

        args, kwargs = replace_argument(
            func=self.model.fit,
            name="callbacks",
            item=self.create_callbacks(
                use_tqdm,
                tqdm_class,
                use_tensorboard,
                tensorboard_dir,
                save_best_after,
                save_filepath,
                additional_callbacks=[],
            ),
            args=args,
            kwargs=kwargs,
            append=True,
        )

        return self.model.fit(*args, **kwargs)

    def reset_weights(self):
        """Reset the model as if you've built a new model."""
        initializers.reinitialize_model_weights(self.model)
