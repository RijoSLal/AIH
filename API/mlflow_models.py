import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import mlflow # type: ignore 
import logging

logger = logging.getLogger() 

logger.setLevel(logging.DEBUG)

class Server_models: 
    """
    A class for retrieving TensorFlow models from a remote MLflow model registry server.
    """

    def __init__(self) -> None:
        """
        Initializes the Server_models class by setting the MLflow tracking URI to the remote server.
        """
        self.track_uri="https://dagshub.com/slalrijo2005/AIH.mlflow"
        mlflow.set_tracking_uri(self.track_uri)


    def model_retriever(self,model_name: str,version: str) -> tf.keras:
        """
        Retrieves a TensorFlow model from the MLflow model registry using its name and version.

        Args:
            model_name (str): The name of the model registered in MLflow.
            version (str or int): The version number of the model.

        Returns:
            A loaded TensorFlow model object.
        """
        model_uri = f"models:/{model_name}/{version}"
        model = self.model_server_retriever(model_uri)
        return model

    def model_server_retriever(self,uri: str) -> tf.keras:
        """
        Loads a TensorFlow model from the specified MLflow model URI.

        Args:
            uri (str): The full MLflow URI of the model to be loaded.

        Returns:
            A loaded TensorFlow model object.
        """
        model = mlflow.tensorflow.load_model(uri)
        return model


