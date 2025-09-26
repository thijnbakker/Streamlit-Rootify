import pytest
from unittest.mock import patch, Mock
from src.npecage.models.model import load_trained_model


@patch("src.npecage.models.model.load_model")
@patch("src.npecage.models.model.f1", autospec=True)
def test_load_trained_model_success(mock_f1, mock_keras_load_model):
    # Arrange
    mock_model = Mock()
    mock_keras_load_model.return_value = mock_model
    model_path = "../../models/model-best.h5"

    # Act
    loaded_model = load_trained_model(model_path)

    # Assert
    mock_keras_load_model.assert_called_once_with(model_path, custom_objects={"f1": mock_f1})
    assert loaded_model == mock_model


@patch("src.npecage.models.model.load_model", side_effect=IOError("File not found"))
@patch("src.npecage.models.model.f1", autospec=True)
def test_load_trained_model_failure(mock_f1, mock_keras_load_model):
    with pytest.raises(IOError, match="File not found"):
        load_trained_model("invalid_model_path.h5")
